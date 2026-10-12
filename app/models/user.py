from app import db
from bson.objectid import ObjectId
import bcrypt

class User:
    collection = db['users']

    @staticmethod
    def create_user(data):
        # Hash password
        if 'password' in data:
            data['password'] = bcrypt.hashpw(data['password'].encode('utf-8'), bcrypt.gensalt())
        return User.collection.insert_one(data)

    @staticmethod
    def get_by_email(email):
        return User.collection.find_one({"email": email})

    @staticmethod
    def get_by_id(user_id):
        try:
            return User.collection.find_one({"_id": ObjectId(user_id)})
        except:
            return None

    @staticmethod
    def verify_password(plain_password, hashed_password):
        return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password)

    @staticmethod
    def update_user(user_id, data):
        return User.collection.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": data}
        )
