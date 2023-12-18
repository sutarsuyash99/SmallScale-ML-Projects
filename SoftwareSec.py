import string
import itertools
import hashlib
from tqdm import tqdm
import threading


# string_letters = []
# string_letters[:] = string.ascii_lowercase + string.ascii_uppercase + string.digits

def find_plain_text(s1, s2, query_hash):
    for x in tqdm(range(len(s1)), total = len(s1)):
    # for x in range(len(s1)):
        y = itertools.product(s2, repeat = 5)
        for z in y:
            plain_text = s1[x] + ''.join(z)
            cipher_text = hashlib.sha256(plain_text.encode("ascii")).hexdigest()
            if cipher_text == query_hash:
                print("\nPlain Text: " + str(plain_text))
                exit()
    print(str(x) + "Completed")


if __name__=="__main__":

    letters = []
    letters[:] = string.ascii_lowercase + string.ascii_uppercase + string.digits 

    query = "8945e8926f0932a931bd9e221c2521f245ad24190cf4f9bdbfa96a496133df01s"

    thread = []
    count = 0
    for i in range(4):
        t1 = threading.Thread(target=find_plain_text, args=(letters[count:count+10], letters, query))
        thread.append(t1)
        count += 10

    t1 = threading.Thread(target=find_plain_text, args=(letters[count:count+11], letters, query))
    thread.append(t1)
    count += 11

    t1 = threading.Thread(target=find_plain_text, args=(letters[count:count+11], letters, query))
    thread.append(t1)
    count += 11

    for t in thread:
        t.start()
    
    for t in thread:
        t.join()

    print("done")