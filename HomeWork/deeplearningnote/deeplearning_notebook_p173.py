# 클래스(상속, 오버라이드,슈퍼)
'''
기존의 클래스를 바탕으로 메서드나 멤버를 추가하거나 일부만 변경하여 새로운 클래스를 생성가능
바탕이 되는 클래스는 부모 클래스, 슈퍼 클래스, 기저 클래스등으로 부르고, 새로 만든 클래스는
자식 클래스, 서브 클래스, 파생 클래스 등으로 부름
    자식 클래스의 경우
- 부모 클래스의 메서드와 멤버를 그대로 사용가능
- 부모 클래스의 메서드와 멤버를 덮어쓸 수 있음(오버드라이브)
- 자기 자신의 메서드와 멤버를 자유롭게 추가할 수 있다.
- 자식 클래스에서 부모 클래스의 메서드와 멤버를 호출할 수 있다.(슈퍼)
'''

# # MyProduct 클래스를 상속하는 MyProductSalesTax을 정의한다.
# class MyProductSalesTax(MyProuduct):
#     # MyProductSalesTax는 생성자의 네 번째 인수가 소비세율을 받는다
#     def __init__(self, name, price, stock, tax_rate):
#         # super()를 사용하면 부모 클래스의 메서드를 호출
#         # 여기서는 MyProduct 클래스의 생성자를 호출
#         super().__init__(name, price,stock)
#         self.tax_rate = tax_rate

#     # MyProductSalesTax에서 MyProduct의 get_name을 재정의(오버라이드)합니다
#     def get_name(self):
#         return self.name + "(소비세 포함)"

#     #MyProductSalesTax에서 get_price_with_tax를 새로 구현합니다
#     def get_price_with_tax(self):
#         return int(self.price * (1 + self.tax_rate))


# product_3 = MyProductSalesTax("phone", 30000, 100, 0.1)
# print(product_3.get_name())
# print(product_3.get_price_with_tax())
# # MyProductSalesTax 클래스에는 summary() 메서드가 정의되어 있지 않지만,
# # MyProduct를 상속하고 있기 때문에 MyProduct의 summary() 메서드를 호출할 수 있다
# product_3.summary()
