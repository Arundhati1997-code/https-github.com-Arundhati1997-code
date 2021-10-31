class Student {
    private int studentID;
    private String name;
    private float mark;
    private char grade;

public Student(int theStudentID, String theName)
{
    studentID = theStudentID;
    name = theName;

}

public void setStudentID(int theStudentID)
{
    studentID = theStudentID;
}

public void setName(String theName)
{
    name = theName;

}

public void setMark(float theMark)
{
    mark = theMark;
}

 public void setGrade(char theGrade) {

    grade = theGrade;
}



public int getStudentID()
{
    return studentID;
}

public String getName()
{
    return name;
}

public float getMark()
{
    return mark;
}


public char getGrade()
{


        if((mark > 70) && (mark <= 100))
                                grade = 'A';
        else if ((mark > 60) && (mark <= 70))
                                grade = 'B';
        else if ((mark > 50) && (mark <= 60))
                                grade = 'C';
        else if ((mark > 40) && (mark <= 50))
                                grade = 'D';
        else

        grade = 'E';

        return grade;
}


public void display()
{
    System.out.println("Student ID is " + studentID);
    System.out.println("Student name is " + name);
    System.out.println("Student mark is " + mark);
    System.out.println("Student grade is " + grade);
}

public static void main (String [] args)
    {

        Student student1 = new Student(1, "mary Dee");
        Student student2 = new Student(2, "john doo");
        Student student3 = new Student(3, "Bart Bloggs");

        student1.setMark(90.4f);
        student2.setMark(55.0f);
        student3.setMark(73.4f);
         student1.getGrade();
         student2.getGrade();
         student3.getGrade();


        student1.display();
        student2.display();
        student3.display();
    }

}
