import torch
'''
# 환경 설정
# conda create -n toto python==3.7 anaconda
1. python == 3.7.0
2. cuda == 10.0
3. cudnn == 10.0

# conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
=> pytorch == 1.6.0 (cuda 10.1 version)

# C++이 설치가 안되어 있어서 import시 errror 뜸
-> error 메세지에 뜬 주소에서 C++다운.설치 후 실행시 정상 작동 함
'''
print(torch.cuda.get_device_name(0))    # GeForce RTX 2080

print(torch.cuda.is_available())        # True : cuda 사용 여부

print(torch.__version__)                # 1.6.0+cu101