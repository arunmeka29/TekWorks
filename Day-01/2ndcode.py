amount = int(input("Enter withdraw amount: "))

five = 0
two = 0
one = 0

if amount % 100 == 0:
    if amount >= 500:
        five = amount // 500
        amount = amount - (five * 500)
        print("500 ---", five)

    if amount >= 200:
        two = amount // 200
        amount = amount - (two * 200)
        print("200 ---", two)

    if amount >= 100:
        one = amount // 100
        amount = amount - (one * 100)
        print("100 ---", one)
else:
    print("Please enter a valid amount")
