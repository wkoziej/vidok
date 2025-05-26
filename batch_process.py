#!/usr/bin/env python3
"""
Batch processor for FramePack video generation.
Reads JSON config and runs multiple demo_gradio.py CLI processes in parallel.
"""

import os
import sys
import json
import argparse
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def load_json(file_path):
    """Load JSON file, return empty dict if file doesn't exist or is invalid"""
    if not os.path.exists(file_path):
        return {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, Exception) as e:
        print(f"Warning: Could not load {file_path}: {e}")
        return {}


def save_json(file_path, data):
    """Save data to JSON file"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_progress_file(config_file):
    """Get progress file path based on config file name"""
    config_path = Path(config_file)
    return config_path.parent / f"{config_path.stem}_progress.json"


def get_failed_file(config_file):
    """Get failed jobs file path based on config file name"""
    config_path = Path(config_file)
    return config_path.parent / f"{config_path.stem}_failed.json"


def create_job_key(job):
    """Create unique key for job based on image path"""
    return job.get('image', '')


def run_single_job(job, output_dir=None, demo_script='demo_gradio.py', gpu_id=None):
    """Run single demo_gradio.py CLI job"""
    image_path = job.get('image')
    prompt = job.get('prompt')
    
    if not image_path or not prompt:
        return False, "Missing image or prompt"
    
    if not os.path.exists(image_path):
        return False, f"Image file not found: {image_path}"
    
    # Build command
    cmd = [
        sys.executable, demo_script, '--cli',
        '--input_image', image_path,
        '--prompt', prompt
    ]
    
    # Add optional parameters
    if job.get('negative_prompt'):
        cmd.extend(['--negative_prompt', job['negative_prompt']])
    if job.get('seed'):
        cmd.extend(['--seed', str(job['seed'])])
    if job.get('duration'):
        cmd.extend(['--length', str(job['duration'])])
    if job.get('steps'):
        cmd.extend(['--steps', str(job['steps'])])
    if job.get('cfg'):
        cmd.extend(['--cfg', str(job['cfg'])])
    if job.get('distilled_cfg'):
        cmd.extend(['--distilled_cfg', str(job['distilled_cfg'])])
    if job.get('cfg_rescale'):
        cmd.extend(['--cfg_rescale', str(job['cfg_rescale'])])
    if job.get('gpu_memory'):
        cmd.extend(['--gpu_memory', str(job['gpu_memory'])])
    if job.get('no_teacache'):
        cmd.append('--no_teacache')
    if job.get('mp4_crf'):
        cmd.extend(['--mp4_crf', str(job['mp4_crf'])])
    
    # Custom output path
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        image_name = Path(image_path).stem
        output_path = os.path.join(output_dir, f"{image_name}.mp4")
        cmd.extend(['--output', output_path])
    elif job.get('output'):
        cmd.extend(['--output', job['output']])
    
    # Set up environment with specific GPU
    env = os.environ.copy()
    if gpu_id is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        gpu_info = f" (GPU {gpu_id})"
    else:
        gpu_info = ""
    
    print(f"🚀 Starting: {os.path.basename(image_path)}{gpu_info}")
    print(f"   Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, env=env)  # 1 hour timeout
        
        if result.returncode == 0:
            print(f"✅ Completed: {os.path.basename(image_path)}{gpu_info}")
            return True, "Success"
        else:
            error_msg = result.stderr.strip() or result.stdout.strip() or "Unknown error"
            print(f"❌ Failed: {os.path.basename(image_path)}{gpu_info} - {error_msg}")
            return False, error_msg
            
    except subprocess.TimeoutExpired:
        print(f"⏰ Timeout: {os.path.basename(image_path)}{gpu_info}")
        return False, "Timeout after 1 hour"
    except Exception as e:
        print(f"💥 Error: {os.path.basename(image_path)}{gpu_info} - {str(e)}")
        return False, str(e)


def process_jobs(config_file, parallel=1, output_dir=None, force_reprocess=False, dry_run=False, demo_script='demo_gradio.py', num_gpus=1):
    """Main processing function"""
    
    # File paths
    progress_file = get_progress_file(config_file)
    failed_file = get_failed_file(config_file)
    
    print(f"📋 Config: {config_file}")
    print(f"📊 Progress: {progress_file}")
    print(f"❌ Failed: {failed_file}")
    print(f"🔧 Parallel: {parallel}")
    print(f"🎮 GPUs: {num_gpus}")
    if output_dir:
        print(f"📁 Output: {output_dir}")
    print("=" * 50)
    
    # Load initial data
    progress_data = {} if force_reprocess else load_json(progress_file)
    failed_data = load_json(failed_file)
    
    processed_count = 0
    failed_count = 0
    
    while True:
        # Reload config (might be updated)
        config_data = load_json(config_file)
        jobs = config_data.get('jobs', [])
        
        if not jobs:
            print("No jobs found in config file")
            break
        
        # Find pending jobs
        pending_jobs = []
        for job in jobs:
            job_key = create_job_key(job)
            if job_key and job_key not in progress_data:
                pending_jobs.append(job)
        
        if not pending_jobs:
            print("🎉 All jobs completed!")
            break
        
        print(f"\n📋 Found {len(pending_jobs)} pending jobs")
        
        if dry_run:
            print("🔍 DRY RUN - would process:")
            for job in pending_jobs:
                print(f"  - {job.get('image')} -> {job.get('prompt')[:50]}...")
            break
        
        # Process jobs in parallel
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            # Submit jobs
            future_to_job = {}
            for i, job in enumerate(pending_jobs):
                # Distribute jobs across available GPUs
                gpu_id = i % num_gpus if num_gpus > 1 else None
                future = executor.submit(run_single_job, job, output_dir, demo_script, gpu_id)
                future_to_job[future] = job
            
            # Process completed jobs
            for future in as_completed(future_to_job):
                job = future_to_job[future]
                job_key = create_job_key(job)
                
                try:
                    success, message = future.result()
                    
                    if success:
                        # Mark as completed
                        progress_data[job_key] = {
                            'image': job.get('image'),
                            'completed_at': time.time(),
                            'status': 'success'
                        }
                        processed_count += 1
                        
                        # Remove from failed if it was there
                        if job_key in failed_data:
                            del failed_data[job_key]
                            save_json(failed_file, failed_data)
                    else:
                        # Mark as failed
                        failed_data[job_key] = {
                            'image': job.get('image'),
                            'failed_at': time.time(),
                            'error': message,
                            'job': job
                        }
                        failed_count += 1
                        save_json(failed_file, failed_data)
                    
                    # Save progress after each job
                    save_json(progress_file, progress_data)
                    
                except Exception as e:
                    print(f"💥 Unexpected error processing {job.get('image')}: {e}")
        
        # Small delay before checking for new jobs
        time.sleep(1)
    
    print(f"\n📊 Final Summary:")
    print(f"  ✅ Processed: {processed_count}")
    print(f"  ❌ Failed: {failed_count}")
    print(f"  📁 Total completed: {len(progress_data)}")


def main():
    parser = argparse.ArgumentParser(description='Batch process FramePack video generation')
    parser.add_argument('--config', '-c', type=str, required=True,
                       help='JSON config file path')
    parser.add_argument('--parallel', '-p', type=int, default=1,
                       help='Number of parallel processes (default: 1)')
    parser.add_argument('--output-dir', '-o', type=str,
                       help='Output directory for videos (optional)')
    parser.add_argument('--force-reprocess', action='store_true',
                       help='Reprocess all jobs, ignore progress file')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be processed without running')
    parser.add_argument('--demo-script', type=str, default='demo_gradio.py',
                       help='Path to demo_gradio.py script (default: demo_gradio.py)')
    parser.add_argument('--num-gpus', type=int, default=1,
                       help='Number of GPUs to use (default: 1)')
    
    args = parser.parse_args()
    
    # Validate config file
    if not os.path.exists(args.config):
        print(f"❌ Error: Config file '{args.config}' not found!")
        sys.exit(1)
    
    # Validate demo_gradio.py exists
    if not os.path.exists(args.demo_script):
        print(f"❌ Error: Demo script '{args.demo_script}' not found!")
        print(f"💡 Try: --demo-script FramePack/demo_gradio.py")
        sys.exit(1)
    
    print("🎬 FramePack Batch Processor")
    print("=" * 50)
    
    try:
        process_jobs(
            config_file=args.config,
            parallel=args.parallel,
            output_dir=args.output_dir,
            force_reprocess=args.force_reprocess,
            dry_run=args.dry_run,
            demo_script=args.demo_script,
            num_gpus=args.num_gpus
        )
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 