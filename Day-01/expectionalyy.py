a=int(input())
b=int(input())
try:
    print(a/b)
except ZeroDivisionError:
    print("Division by zero is not allowed")
except ValueError:
    print("Invalid input")
finally:    
    print("Execution completed")
