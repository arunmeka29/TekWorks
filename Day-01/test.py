project = int(input("Enter project marks: "))
internal = int(input("Enter internal marks: "))
external = int(input("Enter external marks: "))

if project >= 50 and internal >= 50 and external >= 50:
    
    total_marks = (project * 0.7) + (internal * 0.1) + (external * 0.2)

    if total_marks >= 90:
        grade = "A"
    elif total_marks >= 70:
        grade = "B"
    elif total_marks >= 50:
        grade = "C"
    print("Total Marks:", total_marks)
    print("Grade:", grade)
else:
    if project<50:
        print("you got failed in project and score is",project)
    if internal<50:
        print("you got failed in internal and score is",internal)
    if external<50:
         print("you got failed in external and score is",external)

         
        
        

