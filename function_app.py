import logging
import azure.functions as func
import numpy as np
import cv2
from PIL import Image
import io
import base64
import json
import torch
import onnxruntime as ort
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
from skimage import measure, morphology
import os
import tempfile
import traceback

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

# Global variables for models
device = torch.device("cpu")
processor = None
model = None
mlsd_session = None

def initialize_models():
    global processor, model, mlsd_session
    if processor is None or model is None or mlsd_session is None:
        logging.info("Initializing models...")
        
        # Log current working directory and its contents
        cwd = os.getcwd()
        logging.info(f"Current working directory: {cwd}")
        logging.info(f"Contents of current directory: {os.listdir(cwd)}")
        
        # Set and log a temporary directory for cache
        temp_dir = tempfile.gettempdir()
        logging.info(f"Using temporary directory: {temp_dir}")
        os.environ['TRANSFORMERS_CACHE'] = temp_dir
        
        try:
            #processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_ade20k_swin_tiny", cache_dir=temp_dir)
            #model = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_ade20k_swin_tiny", cache_dir=temp_dir).to(device)
            processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_ade20k_swin_tiny", cache_dir=temp_dir)
            model = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_ade20k_swin_tiny", cache_dir=temp_dir, torch_dtype=torch.float32).to(device)
            
            logging.info("OneFormer models loaded successfully")
        except Exception as e:
            logging.error(f"Error loading OneFormer models: {str(e)}")
            raise
        
        # M-LSD model path
        WEIGHT_PATH = os.path.join(cwd, 'M-LSD_512_large.opt.onnx')
        
        logging.info(f"M-LSD Weight Path: {WEIGHT_PATH}")
        
        try:
            mlsd_session = ort.InferenceSession(WEIGHT_PATH)
            logging.info("M-LSD model loaded successfully")
        except Exception as e:
            logging.error(f"Error loading M-LSD model: {str(e)}")
            raise

        logging.info("All models initialized successfully.")

def pred_lines(image, session, input_shape=[512, 512], score_thr=0.22, dist_thr=25.0):
    h, w, _ = image.shape
    h_ratio, w_ratio = [h / input_shape[0], w / input_shape[1]]

    resized_image = cv2.resize(image, (input_shape[0], input_shape[1]), interpolation=cv2.INTER_AREA)
    resized_image = np.concatenate([resized_image, np.ones([input_shape[0], input_shape[1], 1])], axis=-1)
    batch_image = np.expand_dims(resized_image, axis=0).astype('float32')

    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    output = session.run(output_names, {input_name: batch_image})

    pts, pts_score, vmap = output

    start = vmap[0, :, :, :2]
    end = vmap[0, :, :, 2:]
    dist_map = np.sqrt(np.sum((start - end) ** 2, axis=-1))

    segments_list = []
    for center, score in zip(pts[0], pts_score[0]):
        center = center.astype('int32')
        y, x = center
        distance = dist_map[y, x]
        if score > score_thr and distance > dist_thr:
            disp_x_start, disp_y_start, disp_x_end, disp_y_end = vmap[0, y, x, :]
            x_start = x + disp_x_start
            y_start = y + disp_y_start
            x_end = x + disp_x_end
            y_end = y + disp_y_end
            segments_list.append([x_start, y_start, x_end, y_end])

    lines = 2 * np.array(segments_list)  # 256 > 512
    lines[:, 0] = lines[:, 0] * w_ratio
    lines[:, 1] = lines[:, 1] * h_ratio
    lines[:, 2] = lines[:, 2] * w_ratio
    lines[:, 3] = lines[:, 3] * h_ratio

    return lines

def process_image_and_generate_final_mask(image, wall_mask):
    # Detect lines using M-LSD
    lines = pred_lines(image, mlsd_session, input_shape=[512, 512])

    # Create a binary mask
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Draw detected lines on the mask
    for line in lines:
        x1, y1, x2, y2 = map(int, line)
        cv2.line(mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, 2)

    # Ensure the mask is a numpy array with the correct data type
    mask = np.array(mask, dtype=np.uint8)

    # Make sure wall_mask is also a numpy array with the same data type
    wall_mask = np.array(wall_mask, dtype=np.uint8)

    # Perform the subtraction
    result_mask = cv2.subtract(wall_mask, mask)

    # Convert the boolean mask to an unsigned 8-bit integer format (0s and 255s)
    result_mask_uint8 = (result_mask * 255).astype(np.uint8)

    # Invert the mask (black becomes white and vice versa)
    inverted_mask = cv2.bitwise_not(result_mask_uint8)

    # Define the dilation kernel size (adjust to grow by 2 to 3 pixels)
    kernel_size = 2
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Apply dilation to the inverted mask
    dilated_mask = cv2.dilate(inverted_mask, kernel, iterations=1)

    # Invert the mask back to the original form
    final_result_mask = cv2.bitwise_not(dilated_mask)

    return final_result_mask, mask

def get_mask_segments(final_result, drop_percentage=2.0):
    thresholded_mask = (final_result > 128).astype(np.uint8)
    labels = measure.label(thresholded_mask)
    total_pixels = np.prod(final_result.shape)
    min_size = total_pixels * (drop_percentage / 100.0)
    if drop_percentage > 0:
        labeled_mask = morphology.remove_small_objects(labels, min_size=min_size)
    else:
        labeled_mask = labels
    unique_labels = np.unique(labeled_mask)
    segments = [np.where(labeled_mask == label, 1, 0).astype(np.uint8) for label in unique_labels if label > 0]
    return segments

def process_and_segment(image):
    # Process with OneFormer
    inputs = processor(image, ["semantic"], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_semantic_map = processor.post_process_semantic_segmentation(outputs, target_sizes=[image.shape[:2]])[0]
    
    # Create wall mask (assuming wall label ID is 0)
    wall_mask = (predicted_semantic_map == 0).cpu().numpy()

    # Generate final mask and M-LSD output
    final_result, mlsd_output = process_image_and_generate_final_mask(image, wall_mask)

    # Get mask segments
    mask_segments = get_mask_segments(final_result)

    return image, mask_segments, mlsd_output

@app.route(route="image_segment")
def image_segment(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    try:
        # Initialize models
        initialize_models()

        # Get the image data from the request
        image_data = req.get_body()
        logging.info(f"Received image data of size: {len(image_data)} bytes")
        
        # Convert image data to PIL Image
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        image_np = np.array(image)
        logging.info(f"Image shape: {image_np.shape}")

        # Process the image
        _, mask_segments, _ = process_and_segment(image_np)
        logging.info(f"Number of mask segments: {len(mask_segments)}")

        # Convert results to base64 for JSON serialization
        buffered = io.BytesIO()
        Image.fromarray(image_np).save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        mask_segments_list = [base64.b64encode(mask.tobytes()).decode() for mask in mask_segments]

        # Return the results
        return func.HttpResponse(
            json.dumps({
                'image': img_str,
                'mask_segments': mask_segments_list
            }),
            mimetype="application/json",
            status_code=200
        )
    
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        logging.error(f"Error type: {type(e).__name__}")
        logging.error(f"Error traceback: {traceback.format_exc()}")
        return func.HttpResponse(
            f"An error occurred: {str(e)}",
            status_code=500
        )