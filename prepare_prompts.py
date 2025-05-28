import os
import glob
import base64
import requests
import json
import argparse
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print("Error: OPENAI_API_KEY not found. Please set it in a .env file.")
    exit(1)

def encode_image(image_path):
    """Encode image to base64 for API submission"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def generate_motion_prompt(image_path, prompt_type="general"):
    """Generate a motion-focused prompt for the given image using OpenAI's Vision API or a predefined product prompt."""
    image_filename = os.path.basename(image_path)

    if prompt_type == "product":
        return "The product or camera quickly rotates, showcasing product details from different angles."

    # For "general" prompt type, use OpenAI API
    base64_image = encode_image(image_path)
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    
    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "system",
                "content": """You are an assistant that writes short, motion-focused prompts for animating images with subtle but engaging actions. 
                When the user sends an image, respond with a single, 
                concise prompt describing REALISTIC visual motion that enhances the scene without completely transforming it.
                
                IDENTIFY the main subject(s) in the image - this could be: people, animals, objects, vehicles, robots, 
                landscapes, buildings, food, plants, or any other prominent element - and animate THAT subject.
                
                PREFER: Natural and engaging movements appropriate to the subject type. Try to tell simple and one sentence story about the scene.
                
                AVOID: Static poses, and sounds
                
                Think: What would make the MAIN SUBJECT more alive, engaging and active ?
                Examples of preferred styles:
                - "The cat is jumping around the room catching the mouse"
                - "The old man is walking slowly around the room thinking about his life"
                - "The dancing woman is dancing gracefully"

                
                Stay in a loop: one image in, one engaging motion prompt out. 
                Do not explain, ask questions, or generate multiple options."""
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300
    }
    
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        prompt = result["choices"][0]["message"]["content"].strip()
        return prompt
    else:
        print(f"Error processing {image_filename}: {response.text}")
        return None

def load_existing_data(output_file):
    """Load existing JSON data if file exists"""
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Convert to dict for faster lookup
                existing = {}
                for item in data.get('jobs', []):
                    if 'image' in item:
                        existing[item['image']] = item
                return existing
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not parse existing file {output_file}: {e}")
            return {}
    return {}

def clean_missing_images(existing_data, image_dir, dry_run=False):
    """Remove entries from existing_data where the image file doesn't exist"""
    cleaned_data = {}
    removed_count = 0
    
    for image_path, entry in existing_data.items():
        # Check if the image file exists
        # Handle both absolute and relative paths
        if os.path.isabs(image_path):
            full_path = image_path
        else:
            full_path = os.path.join(os.getcwd(), image_path)
        
        if os.path.exists(full_path):
            cleaned_data[image_path] = entry
        else:
            if dry_run:
                print(f"  🗑️  Would remove entry for missing image: {image_path}")
                # In dry run, keep the entry in cleaned_data to avoid affecting the rest of the logic
                cleaned_data[image_path] = entry
            else:
                print(f"  🗑️  Removing entry for missing image: {image_path}")
            removed_count += 1
    
    return cleaned_data, removed_count

def save_data(output_file, data_dict):
    """Save data to JSON file"""
    # Convert dict back to list format
    jobs_list = list(data_dict.values())
    
    output_data = {
        "jobs": jobs_list
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser(description='Generate motion prompts for images and save to JSON')
    parser.add_argument('--images', '-i', type=str, required=True,
                       help='Directory containing images')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Output JSON file path')
    parser.add_argument('--duration', '-d', type=float, default=5.0,
                       help='Default video duration in seconds (default: 5.0)')
    parser.add_argument('--seed', '-s', type=int, default=31337,
                       help='Default seed value (default: 31337)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without making any changes')
    parser.add_argument('--prompt-type', type=str, choices=['general', 'product'], default='general',
                        help='Type of prompt to generate (default: general)')
    
    args = parser.parse_args()
    
    image_dir = args.images
    output_file = args.output
    default_duration = args.duration
    default_seed = args.seed
    dry_run = args.dry_run
    prompt_type_arg = args.prompt_type
    
    # Check if image directory exists
    if not os.path.exists(image_dir):
        print(f"Error: Image directory '{image_dir}' does not exist!")
        return
    
    # Get all images in the directory
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
    
    if not image_paths:
        print(f"No images found in {image_dir}")
        return
    
    print(f"Found {len(image_paths)} images in {image_dir}")
    
    if dry_run:
        print(f"🔍 DRY RUN MODE - No changes will be made")
    
    # Load existing data
    existing_data = load_existing_data(output_file)
    print(f"Loaded {len(existing_data)} existing entries from {output_file}")
    
    # Clean up entries for missing images
    removed_count = 0  # Track removed entries for summary
    if existing_data:
        print(f"🧹 Checking for missing images...")
        existing_data, removed_count = clean_missing_images(existing_data, image_dir, dry_run)
        if removed_count > 0:
            if dry_run:
                print(f"  Would remove {removed_count} entries for missing images")
            else:
                print(f"  Removed {removed_count} entries for missing images")
                # Save the cleaned data immediately
                save_data(output_file, existing_data)
                print(f"  Updated {output_file}")
        else:
            print(f"  No missing images found")
    
    # Process images
    new_count = 0
    skipped_count = 0
    
    for i, image_path in enumerate(image_paths):
        # Use relative path for consistency
        relative_image_path = os.path.relpath(image_path)
        image_filename = os.path.basename(image_path)
        
        print(f"\nProcessing image {i+1}/{len(image_paths)}: {image_filename}")
        
        # Check if this image already has a prompt
        if relative_image_path in existing_data:
            print(f"  ✓ Already processed, skipping")
            skipped_count += 1
            continue
        
        # Generate prompt for new image
        if dry_run:
            print(f"  🤖 Would generate prompt for this image (type: {prompt_type_arg})")
            new_count += 1
        else:
            print(f"  🤖 Generating prompt (type: {prompt_type_arg})...")
            prompt = generate_motion_prompt(image_path, prompt_type_arg)
            
            if prompt:
                # Add new entry
                existing_data[relative_image_path] = {
                    "image": relative_image_path,
                    "prompt": prompt,
                    "duration": default_duration,
                    "seed": default_seed
                }
                print(f"  ✓ Generated: {prompt}")
                new_count += 1
                
                # Save after each successful generation (in case of interruption)
                save_data(output_file, existing_data)
            else:
                print(f"  ❌ Failed to generate prompt")
    
    print(f"\n📊 Summary:")
    print(f"  Total images: {len(image_paths)}")
    print(f"  Already processed: {skipped_count}")
    if dry_run:
        print(f"  Would process: {new_count}")
        if new_count == 0:
            print(f"  Failed: 0")
        if removed_count > 0:
            print(f"  Would remove entries for missing images: {removed_count}")
        print(f"  No changes made (dry run)")
    else:
        print(f"  Newly processed: {new_count}")
        print(f"  Failed: {len(image_paths) - skipped_count - new_count}")
        if removed_count > 0:
            print(f"  Removed entries for missing images: {removed_count}")
        print(f"  Output saved to: {output_file}")

if __name__ == "__main__":
    main() 