from inference_sdk import InferenceHTTPClient
import json

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="wMRJCR9vOvj52DZdgHOk"
)

def analyze(image_path):
    try:
        result = client.run_workflow(
            workspace_name="arun-hvf7d",
            workflow_id="custom-workflow",
            images={"image": image_path},
            use_cache=True
        )

        print("RAW RESULT:", json.dumps(result, indent=2) if isinstance(result, (dict, list)) else result)
        
        predictions = []
        fruit_name = "Unknown"
        confidence = 0
        quality = "Unknown"
        
        if isinstance(result, list) and len(result) > 0:
            first_item = result[0]
            
            if isinstance(first_item, dict) and 'predictions' in first_item:
                predictions_data = first_item['predictions']
                
                if isinstance(predictions_data, dict) and 'predictions' in predictions_data:
                    predictions = predictions_data['predictions']
                    
                    if len(predictions) > 0:
                        pred = predictions[0]
                        class_name = pred.get('class', 'unknown')
                        confidence = pred.get('confidence', 0)
                        
                        class_lower = class_name.lower()
                        
                        if 'apple' in class_lower:
                            fruit_name = 'Apple'
                        elif 'banana' in class_lower:
                            fruit_name = 'Banana'
                        elif 'orange' in class_lower:
                            fruit_name = 'Orange'
                        else:
                            fruit_name = class_name.capitalize()
                        
                        if 'fresh' in class_lower:
                            quality = 'Fresh'
                        elif 'rotten' in class_lower:
                            quality = 'Rotten'
                        else:
                            quality = 'Unknown'
                            
                elif isinstance(predictions_data, list):
                    predictions = predictions_data
                    if len(predictions) > 0:
                        pred = predictions[0]
                        class_name = pred.get('class', 'unknown')
                        confidence = pred.get('confidence', 0)
                        
                        class_lower = class_name.lower()
                        
                        if 'apple' in class_lower:
                            fruit_name = 'Apple'
                        elif 'banana' in class_lower:
                            fruit_name = 'Banana'
                        elif 'orange' in class_lower:
                            fruit_name = 'Orange'
                        else:
                            fruit_name = class_name.capitalize()
                        
                        if 'fresh' in class_lower:
                            quality = 'Fresh'
                        elif 'rotten' in class_lower:
                            quality = 'Rotten'
                        else:
                            quality = 'Unknown'
        
        print("EXTRACTED PREDICTIONS:", predictions)
        print("FRUIT DETECTED:", fruit_name)
        print("QUALITY:", quality)
        print("CONFIDENCE:", confidence)

        if len(predictions) == 0:
            return [], "No Fruit Detected", "Unknown", 0, "Unknown"

        if quality == 'Fresh':
            grade = f"Fresh {fruit_name}"
        elif quality == 'Rotten':
            grade = f"Rotten {fruit_name}"
        else:
            grade = f"{fruit_name} (Quality Unknown)"

        return predictions, grade, fruit_name, confidence, quality
        
    except Exception as e:
        print(f"Error in analyze function: {str(e)}")
        import traceback
        traceback.print_exc()
        return [], f"Error: {str(e)}", "Unknown", 0, "Unknown"