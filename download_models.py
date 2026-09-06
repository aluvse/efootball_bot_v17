from pathlib import Path
from huggingface_hub import snapshot_download

ROOT=Path(__file__).parent
MODELS={
    'rtdetr': ('PekingU/rtdetr_r50vd', ROOT/'models'/'rtdetr_r50vd'),
    'mobilenet': ('timm/mobilenetv3_large_100.miil_in21k_ft_in1k', ROOT/'models'/'mobilenetv3_large'),
}

def main():
    for name,(repo,dest) in MODELS.items():
        dest.mkdir(parents=True,exist_ok=True)
        print(f'[MODEL] downloading {name}: {repo}')
        snapshot_download(repo_id=repo,local_dir=str(dest),local_dir_use_symlinks=False)
        print(f'[MODEL] ready: {dest}')
    print('[MODEL] all models downloaded')
if __name__=='__main__': main()
