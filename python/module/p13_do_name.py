import p11_car
import p12_tv

# import하는 시점에서 순간 한번 훝어주면서 실행
print("=====================")
print("do.py의 module 이름은 ", __name__)   # 실행 시점에서의 이 파일이 main 이라서 #__main__
print("=====================")              # 임포트한 파일의 경우 그 파일의 이름 적용

p11_car.drive()
p12_tv.watch()