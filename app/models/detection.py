from app import db
from bson.objectid import ObjectId
import datetime

class Detection:
    collection = db['detections']

    @staticmethod
    def save_detection(user_id, crop, disease, confidence, image_path):
        record = {
            "user_id": ObjectId(user_id) if user_id else None,
            "crop": crop,
            "disease": disease,
            "confidence": confidence,
            "image_path": image_path,
            "timestamp": datetime.datetime.utcnow()
        }
        result = Detection.collection.insert_one(record)
        return str(result.inserted_id)

    @staticmethod
    def get_detection(detection_id):
        try:
            return Detection.collection.find_one({"_id": ObjectId(detection_id)})
        except:
            return None

    @staticmethod
    def get_user_history(user_id):
        try:
            return list(Detection.collection.find({"user_id": ObjectId(user_id)}).sort("timestamp", -1))
        except:
            return []
