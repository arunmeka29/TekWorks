class Bank:
    acbal = 10000

    def withdraw(self):
        amt = int(input("Enter your withdraw amount: "))

        if amt % 100 != 0:
            print("Please enter multiples of 100 only")
            return

        if amt > 20000:
            print("Withdraw limit is 20000 only")
            return

        if amt > Bank.acbal:
            print("Insufficient fund")
            return

        Bank.acbal -= amt
        print("Please collect your cash")
        print("Available balance:", Bank.acbal)

    def deposit(self):
        amt = int(input("Enter your deposit amount: "))

        if amt % 100 != 0:
            print("Please enter multiples of 100 only")
            return

        Bank.acbal += amt
        print("Deposit successful")
        print("Available balance:", Bank.acbal)

    def viewOptions(self):
        ch = 'y'
        while ch == 'y' or ch == 'Y':

            print("\n1. Deposit")
            print("2. Withdraw")
            print("3. Balance Enquiry")
            print("0. EXIT")

            option = int(input("Choose your option: "))

            if option == 1:
                self.deposit()
            elif option == 2:
                self.withdraw()
            elif option == 3:
                print("Available balance:", Bank.acbal)
            elif option == 0:
                print("Thank you, visit again")
                exit()
            else:
                print("Invalid option")

            ch = input("Do you want to continue? (y/n): ")



obj = Bank()
obj.viewOptions()
