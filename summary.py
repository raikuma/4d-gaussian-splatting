import sys
import json
import collections

if __name__ == "__main__":
    model_path = sys.argv[1]
    iteration = sys.argv[2]

    results = collections.defaultdict(str)

    # metrics
    try:
        with open(f"{model_path}/results.json", 'r') as f:
            metrics = json.load(f)
            values = metrics['ours_None']
            results['full-psnr'] = values['PSNR']
            results['full-ssim'] = values['SSIM']
            results['full-alex'] = values['LPIPS_ALEX']
    except:
        pass
        
    # metrics mask
    try:
        with open(f"{model_path}/results_mask.json", 'r') as f:
            metrics = json.load(f)
            values = metrics['ours_None']
            results['mask-psnr'] = values['PSNR']
            results['mask-ssim'] = values['SSIM']
            results['mask-alex'] = values['ALEX']
    except:
        pass

    # fps
    try:
        with open(f"{model_path}/fps.txt", 'r') as f:
            fps = f.read().strip().split(' ')[-1]
            results['fps'] = fps
    except:
        pass

    # storage
    try:
        with open(f"{model_path}/storage.txt", 'r') as f:
            storage = f.read().strip().split('\n')[-1].split(' ')[-1]
            results['storage'] = storage
    except:
        pass

    # number
    try:
        with open(f"{model_path}/number.txt", 'r') as f:
            lines = f.read().strip().split('\n')
            for line in lines:
                key, value = line.split(': ')
                results[key.lower()] = value
    except:
        pass

    # print results
    print(f"full-PSNR	full-SSIM	full-ALEX	mask-PSNR	mask-SSIM	mask-ALEX	anchor	total	active	static	dynamic	storage	fps")
    print(f"{results['full-psnr']}	{results['full-ssim']}	{results['full-alex']}	"
          f"{results['mask-psnr']}	{results['mask-ssim']}	{results['mask-alex']}	"
          f"{results['anchor']}	{results['total']}	{results['active']}	{results['static']}	{results['dynamic']}	"
          f"{results['storage']}	{results['fps']}")