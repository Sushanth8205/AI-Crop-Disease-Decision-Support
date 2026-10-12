from app import db
from bson.objectid import ObjectId
import datetime

class Appointment:
    collection = db['appointments']

    @staticmethod
    def create_appointment(farmer_id, advisor_id, date, time_slot, report_path=None):
        record = {
            "farmer_id": ObjectId(farmer_id),
            "advisor_id": ObjectId(advisor_id),
            "date": date,
            "time_slot": time_slot,
            "report_path": report_path,
            "status": "scheduled",
            "created_at": datetime.datetime.utcnow()
        }
        result = Appointment.collection.insert_one(record)
        return str(result.inserted_id)

    @staticmethod
    def is_slot_booked(advisor_id, date, time_slot):
        count = Appointment.collection.count_documents({
            "advisor_id": ObjectId(advisor_id),
            "date": date,
            "time_slot": time_slot,
            "status": "scheduled"
        })
        return count > 0

    @staticmethod
    def get_farmer_appointments(farmer_id):
        return list(Appointment.collection.find({"farmer_id": ObjectId(farmer_id)}).sort("date", 1))

    @staticmethod
    def get_advisor_appointments(advisor_id):
        return list(Appointment.collection.find({"advisor_id": ObjectId(advisor_id)}).sort("date", 1))
