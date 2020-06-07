fruits = {"strawberry": "red", "peach" : "pink", "banana": "yellow"}
for fruit, color in fruits.items():
    print(fruit + " is " +  color)
'''
딕셔너리형 루프에서는 키와 값을 모두 변수로 하여 반복(루프)할 수 있습니다. items()를 사용하여
'for key의_변수명, values의_변수명 in 변수(딕셔너리형).items():'로 기술합니다
'''

# 147p 메서드
 
'''
매서드는 어떠한 값에 대해 처리를 하는 것이며, '값, 메서드명()' 형식으로 기술합니다. 역활은 함수와 동일
하지만 함수의 경우 처리하려는 값을 ()안에 기입했지만, 메서드는 값 뒤에 . (점)을 연결해 기술한다는점을 기억!!
ex) append()는 리스트형에 사용할 수 있는 메서드
    함수 
    number = [1,5,3,4,2]
    print(sorted(number))
    print(number)
    
    메서드
    number.sort()
    print(number)
'''

# 문자열형 메서드(upper, count)

'''
upeer()는 모든 문자열을 대문자로 반환하는 메서드입니다. count()는 ()안에 들어 있는 문자열에 요소가 몇 개 포함되어 있는지 알려주는 메서드 입니다.
'''
city = "Tokyo"
print(city.upper()) # TOKYO
print(city.count("o"))  # 2

# 문자열형 메서드 format
'''
format() 메서드는 임의의 값을 대입한 문자열을 생성
'''
print("나는 {}에서 태어나 {}에서 유년기를 보냈습니다".format("서울", "광명시"))

# 클래스
'''
클래스(정의)   - 자동차 설계도
생서장(함수)   - 자동차 공장
멤버(변수)     - 휘발유의 양, 현재 속도 등
메서드(함수)   - 브레이크, 엑셀, 핸들 등
인스턴스(실체) - 공장에서 만들어진 실제 자동차 

생성자(constructor)
- 클래스를 만들 때 자동으로 호출되는 함수. 파이썬에서 이름을 __init__으로 할 필요가 있다. 첫 번째 인수로 객체 자신을 의미하는 self라는 특수한 변수를 갖게 됨

메서드(method)
- 클래스가 갖는 처리, 즉 함수이다. 인스턴스를 조직하는 인스턴스 메서드, 클래스 전체를 처리하는 클래스 메서드, 인스턴스 없이도 실행할 수 있는 정적메서드의 세
종류가 존재.

멤버(member)
- 클래스가 가지는 값, 즉 변수. 다른 객체 지향 언어에는 프라이빗(private, 클래스 외부에서 접근 불가)과 퍼블릭(public, 클래스 외부에서 접근 가능) 두 종류의
멤버가 마련되어 있지만, 파이썬에서는 모두 퍼블릭 멤버로 처리됩니다. 대신 파이썬에서는 멤버에 대한 접근(access)을 (property)로 제한할 수 있다. 
'''

# 클래스( 멤버와 생성자 )

# # MyProduct 클래스를 정의합니다
# class MyProduct:

# # 생성자를 정의합니다
#     def __init__(self, name, price):
#         # 인수를 멤버에 저장합니다.
#         self.name = name
#         self.price = price
#         self.stock = 0
#         self.sales = 0
'''
클래스는 설계도일 뿐 객체를 만드려면 클래스를 호출해야 함
'''
# # MyProduct를 호출하여 객체 product1을 만듭니다.
# product1 = MyProduct("cake", 500)

'''
클래스를 호출할 때 작동하는 메서드를 생성자라고 한다. 생성자는 __init__()로 정의하며, self를 생성자의 첫 번째 인수로 지정해야 한다.
클래스 내 멤버는 self.price 처럼 변수명 앞에 self.을 붙인다. 위의 예에서는 MyProduct가 호출되면 name = "cake",  price =500으로
생성자가 작동하고, 각 인수에 의해 멤버 name, price가 초기화 됩니다. 생성된 객체의 멤버를 참조할 때는 '객체.변수명'으로 직접 참조할
수 있다. 직접 참조에서 멤버의 변경도 가능
'''

# 멤버 변경

class MyProduct:

# 생성자를 정의합니다
    def __init__(self, name, price,stock):
        # 인수를 멤버에 저장합니다.
        self.name = name
        self.price = price
        self.stock = stock
        self.sales = 0

    # 구매 메서드
    def buy_up(self,n):
        self.stock += n

    # 판매 메서드
    def sell(self,n):
        self.stock -=n
        self.sales += n*self.price

    # 개요 메서드
    def summary(self):
        message = "called summary().\n name: " + self.name + \
        "\\ price : " + str(self.price) + \
        "\n stock : " + str(self.stock) + \
        "\n sales : " + str(self.sales)
        print(message)