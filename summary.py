import sys
import json

if __name__ == "__main__":
    model_path = sys.argv[1]
    iteration = sys.argv[2]

    results = {}

    # metrics
    with open(f"{model_path}/results.json", 'r') as f:
        metrics = json.load(f)
        values = metrics['ours_'+iteration]
        results['full-psnr'] = values['PSNR']
        results['full-ssim'] = values['SSIM']
        results['full-alex'] = values['ALEX']
        
    # metrics mask
    with open(f"{model_path}/results_mask.json", 'r') as f:
        metrics = json.load(f)
        values = metrics['ours_'+iteration]
        results['mask-psnr'] = values['PSNR']
        results['mask-ssim'] = values['SSIM']
        results['mask-alex'] = values['ALEX']

    # fps
    with open(f"{model_path}/fps.txt", 'r') as f:
        fps = f.read().strip().split(' ')[-1]
        results['fps'] = fps

    # storage
    with open(f"{model_path}/storage.txt", 'r') as f:
        storage = f.read().strip().split('\n')[-1].split(' ')[-1]
        results['storage'] = storage

    # number
    with open(f"{model_path}/number.txt", 'r') as f:
        lines = f.read().strip().split('\n')
        for line in lines:
            key, value = line.split(': ')
            results[key.lower()] = value

    # print results
    print(f"full-PSNR	full-SSIM	full-ALEX	mask-PSNR	mask-SSIM	mask-ALEX	anchor	total	active	storage	fps")
    print(f"{results['full-psnr']}	{results['full-ssim']}	{results['full-alex']}	"
          f"{results['mask-psnr']}	{results['mask-ssim']}	{results['mask-alex']}	"
          f"{results['anchor']}	{results['total']}	{results['active']}	"
          f"{results['storage']}	{results['fps']}")