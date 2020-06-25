from mechine.car import drive
from mechine.tv import watch

drive()
watch()


from mechine import car
from mechine import tv

car.drive()
tv.watch()


print("==============================")

from mechine.test.car import drive
from mechine.test.tv import watch

drive()
watch()

from mechine.test import car
from mechine.test import tv

car.drive()
tv.watch()

from mechine import test

test.car.drive()
test.tv.watch()

