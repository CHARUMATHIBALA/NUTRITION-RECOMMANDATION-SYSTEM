import bcrypt

# Generate hashed password using bcrypt
password = "password".encode('utf-8')
hashed = bcrypt.hashpw(password, bcrypt.gensalt())
print(hashed.decode('utf-8'))
