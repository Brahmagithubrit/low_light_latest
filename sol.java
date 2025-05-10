import java.util.*;

class Node {
    int data;
    Node left, right;

    Node(int data) {
        this.data = data;
        this.left = null;
        this.right = null;
    }
}
class sol {
    public static void main(String args[]) {
        Node root = new Node(1);
        Node first = new Node(2);
        Node sec = new Node(3);
        Node third = new Node(4);
        Node four = new Node(5);

        root.left = first;
        root.right = sec;
        first.left = third;
        first.right = four;
        ArrayList<
        bottomUp(root , "");
    }

    public static void bottomUp(Node root , String path ) {
        if (root == null)
            return;

        String currPath = root.data + (path.isEmpty() ? "" : "->" + path);
        if (root.left == null || root.right == null) {
            System.out.println(currPath);
            return;
        }
        
        bottomUp(root.left, currPath);
        bottomUp(root.right, currPath);
    }
}