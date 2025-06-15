#include <bits/stdc++.h>

#include <string>

using namespace std;

class Sample {
    public: int a;
    void get(int x) {
        a = x;
    }
    void show() {
        cout << a;
    }

    void operator - () {
        a = -a;
    }
    void operator++() {
        a = a + 1;
    }
};
// operator overloading in containership 
class Container {
    public: int a;

    void setValue(int x) {
        a = x;
    }


};
class OtherClass {
    public: Container cont;

    void setValue(int x) {
        cont.setValue(x);
    }

    void operator++() {
        cont.a = cont.a + 1;
    }
    void display() {
        cout << cont.a;
    }
};
// inheritance with virtual and pure virtual functon with pointer concept 
class Parent {
    public: virtual void show() = 0;
};
class child: public Parent {
    public: void show() {
        cout << "Child";

    }
};
// frined funcitn using scope resolutin operator 
class Random {
    public: int x;
    void display();
    void show() {
        cout << "Working";
    }
    friend void hi(Random & obj);
};
void hi(Random & obj) {
    cout << "hello ji kya hal chal ";
}
// function outside of class 
void Random::display() {
    cout << "Acess done ";
}
// now lets start array of object 
class Random2 {
    public: string name;
    void setName(string name) {
        this -> name = name;
    }
    string show() {
        return  name;
    }
};
int main() {
    Random2 obj[3];
    obj[0].setName("Brahma");
    obj[1].setName("hello");

    // print 
    for (int i = 0; i < 3; i++) {
        cout << obj[i].show();
    }
}