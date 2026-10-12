import os
from flask import Blueprint, request, render_template, redirect, url_for, session, flash, current_app
from werkzeug.utils import secure_filename
from app.models.user import User
from app.models.appointment import Appointment
from bson.objectid import ObjectId

advisor_bp = Blueprint('advisor', __name__)

ALLOWED_EXTENSIONS = {'pdf'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@advisor_bp.route('/list')
def list_advisors():
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))
        
    search_query = request.args.get('search', '').lower()
    
    # Get all advisors
    all_users = User.collection.find({"role": "advisor"})
    
    advisors = []
    for user in all_users:
        # Basic filtering
        if search_query:
            if search_query not in str(user.get('crop_specialization', '')).lower() and \
               search_query not in str(user.get('disease_specialization', '')).lower() and \
               search_query not in str(user.get('district', '')).lower():
                continue
        advisors.append(user)
        
    current_user = User.get_by_id(session['user_id'])
    return render_template('advisor_list.html', advisors=advisors, search_query=search_query, user=current_user)

@advisor_bp.route('/book/<advisor_id>', methods=['GET', 'POST'])
def book(advisor_id):
    if 'user_id' not in session or session.get('role') != 'user':
        flash('Only farmers can book appointments.', 'error')
        return redirect(url_for('main.dashboard'))
        
    advisor = User.get_by_id(advisor_id)
    if not advisor or advisor.get('role') != 'advisor':
        flash('Advisor not found.', 'error')
        return redirect(url_for('advisor.list_advisors'))
        
    if request.method == 'POST':
        date = request.form.get('date')
        time_slot = request.form.get('time_slot')
        
        # Double booking check
        if Appointment.is_slot_booked(advisor_id, date, time_slot):
            flash('This time slot is already booked. Please choose another.', 'error')
            return redirect(url_for('advisor.book', advisor_id=advisor_id))
            
        report_path = None
        if 'report' in request.files:
            file = request.files['report']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                import time
                unique_filename = f"report_{int(time.time())}_{filename}"
                filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], unique_filename)
                file.save(filepath)
                report_path = f"uploads/{unique_filename}"
                
        Appointment.create_appointment(session['user_id'], advisor_id, date, time_slot, report_path)
        flash('Appointment successfully booked!', 'success')
        return redirect(url_for('main.dashboard'))
        
    return render_template('book_appointment.html', advisor=advisor)
