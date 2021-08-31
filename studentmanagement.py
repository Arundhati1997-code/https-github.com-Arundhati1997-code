import csv
# Define global variables
student_fields = ['roll', 'name', 'age', 'email', 'phone']
student_database = 'students.csv'


def display_menu():
    print("--------------------------------------")
    print(" Welcome to Student Management System")
    print("---------------------------------------")
    print("1. Add New Student")
    print("2. View Students")
    print("3. Search Student")
    print("4. Quit")
    
    


def add_student():
    print("-------------------------")
    print("Add Student Information")
    print("-------------------------")
    global student_fields
    global student_database

    student_data = []
    for field in student_fields:
        value = input("Enter " + field + ": ")
        student_data.append(value)

    with open(student_database, "a", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows([student_data])

    print("Data saved successfully")
    input("Press any key to continue")
    return


def view_students():
    global student_fields
    global student_database

    print("--- Student Records ---")

    with open(student_database, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for x in student_fields:
            print(x, end='\t |')
        print("\n-----------------------------------------------------------------")

        for row in reader:
            for item in row:
                print(item, end="\t |")
            print("\n")

    input("Press any key to continue")


def search_student():
    global student_fields
    global student_database

    print("--- Search Student ---")
    roll = input("Enter roll no. to search: ")
    with open(student_database, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) > 0:
                if roll == row[0]:
                    print("----- Student Found -----")
                    print("Roll: ", row[0])
                    print("Name: ", row[1])
                    print("Age: ", row[2])
                    print("Email: ", row[3])
                    print("Phone: ", row[4])
                    break
        else:
            print("Roll No. not found in our database")
    input("Press any key to continue")



while True:
    display_menu()

    choice = input("Enter your choice: ")
    if choice == '1':
        add_student()
    elif choice == '2':
        view_students()
    elif choice == '3':
        search_student()
    else:
        break

print("-------------------------------")
print(" Thank you for using our system")
print("-------------------------------")


Output:
--------------------------------------
 Welcome to Student Management System
---------------------------------------
1. Add New Student
2. View Students
3. Search Student
4. Quit
Enter your choice: 1
-------------------------
Add Student Information
-------------------------
Enter roll: 106
Enter name: Aman chopra
Enter age: 12
Enter email: aman@chopra.com
Enter phone: 202-368-01584
Data saved successfully
Press any key to continue
--------------------------------------
 Welcome to Student Management System
---------------------------------------
1. Add New Student
2. View Students
3. Search Student
4. Quit
Enter your choice: 2
--- Student Records ---
roll	 |name	 |age	 |email	 |phone	 |
-----------------------------------------------------------------
﻿roll	 |name	 |age	 |email	 |phone	 |

101	 |John Doe	 |12	 |john@doe.com	 |202-555-0178	 |

102	 |Sarah Danes	 |13	 |sarah@danes.com	 |202-453-0143	 |

103	 |Siya Sharma	 |14	 |siya@sharma.com	 |202-334-0156	 |

104	 |AryanDoshi	 |15	 |aryan@doshi.com	 |202-667-0166	 |

105	 |Zoya Siddiqui	 |11	 |Zoya@Siddiqui.com	 |202-378-01489	 |

106	 |Aman chopra	 |12	 |aman@chopra.com	 |202-368-01584	 |



Press any key to continue
--------------------------------------
 Welcome to Student Management System
---------------------------------------
1. Add New Student
2. View Students
3. Search Student
4. Quit
Enter your choice: 3
--- Search Student ---
Enter roll no. to search: 102
----- Student Found -----
Roll:  102
Name:  Sarah Danes
Age:  13
Email:  sarah@danes.com
Phone:  202-453-0143
Press any key to continue
--------------------------------------
 Welcome to Student Management System
---------------------------------------
1. Add New Student
2. View Students
3. Search Student
4. Quit
Enter your choice: 4
-------------------------------
 Thank you for using our system
-------------------------------
>>> 



Csv file data:
  roll	name	age	email	phone
101	John Doe	12	john@doe.com	202-555-0178
102	Sarah Danes	13	sarah@danes.com	202-453-0143
103	Siya Sharma	14	siya@sharma.com	202-334-0156
104	AryanDoshi	15	aryan@doshi.com	202-667-0166
105	Zoya Siddiqui	11	Zoya@Siddiqui.com	202-378-01489
106	Aman chopra	12	aman@chopra.com	202-368-01584
