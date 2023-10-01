class Node(object):
    def __init__(self, key, data=None):
        self.key = key
        self.children = {}

class Trie:
    def __init__(self):
        self.head = Node(None)

    def insert(self, string):
        current_node = self.head

        for char in string:
            if char not in current_node.children:
                current_node.children[char] = Node(char)
            current_node = current_node.children[char]


    def search(self, string):
        current_node = self.head

        for char in string:
            if char in current_node.children:
                current_node = current_node.children[char]
            else:
                return False
        return True
    
if __name__ == "__main__":
    input_text_list = ["풍경 묘사", "풍경 묘사 해줘", "풍경 묘사 좀 부탁해", "객체인식 좀 해줘"]
    for input_text in input_text_list:
        word_list = input_text.split()
        MyTrie = Trie()
        for word in word_list:
            MyTrie.insert(word)
        print(MyTrie.search("풍경") or MyTrie.search("묘사"),end=' ')
        print(MyTrie.search("객체") or MyTrie.search("인식"))