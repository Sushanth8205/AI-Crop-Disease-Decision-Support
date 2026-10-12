from flask import Blueprint, render_template, session, redirect, url_for, request, flash
from app.models.user import User
from app.models.appointment import Appointment
from app.models.detection import Detection

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    return render_template('index.html', active_tab=None)

@main_bp.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))

    user_data = User.get_by_id(session['user_id'])
    if not user_data:
        session.clear()
        return redirect(url_for('auth.login'))

    role = session.get('role', 'user')
    active_tab = request.args.get('tab', 'home')
    appointments = []
    history = []

    if role == 'user':
        appointments = Appointment.get_farmer_appointments(session['user_id'])
        for appt in appointments:
            advisor = User.get_by_id(appt['advisor_id'])
            appt['advisor_name'] = advisor['name'] if advisor else 'Unknown'
        history = Detection.get_user_history(session['user_id'])

    elif role == 'advisor':
        appointments = Appointment.get_advisor_appointments(session['user_id'])
        for appt in appointments:
            farmer = User.get_by_id(appt['farmer_id'])
            appt['farmer_name'] = farmer['name'] if farmer else 'Unknown'
            appt['farmer_location'] = f"{farmer.get('location', '')}, {farmer.get('district', '')}, {farmer.get('state', '')}".strip(', ') if farmer else ''
            appt['farmer_lat'] = farmer.get('lat') if farmer else None
            appt['farmer_lng'] = farmer.get('lng') if farmer else None

    return render_template(
        'dashboard.html',
        user=user_data,
        role=role,
        active_tab=active_tab,
        appointments=appointments,
        history=history
    )

@main_bp.route('/profile/update', methods=['GET', 'POST'])
def update_profile():
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))

    user_data = User.get_by_id(session['user_id'])
    if not user_data:
        session.clear()
        return redirect(url_for('auth.login'))

    if request.method == 'POST':
        update_data = {
            'name': request.form.get('name'),
            'phone': request.form.get('phone'),
            'location': request.form.get('location'),
            'district': request.form.get('district'),
            'state': request.form.get('state'),
            'lat': request.form.get('lat'),
            'lng': request.form.get('lng'),
        }

        if user_data.get('role') == 'advisor':
            update_data.update({
                'experience': request.form.get('experience'),
                'qualification': request.form.get('qualification'),
                'crop_specialization': request.form.get('crop_specialization'),
                'disease_specialization': request.form.get('disease_specialization'),
                'available_days': request.form.get('available_days'),
                'available_time_slots': request.form.get('available_time_slots'),
                'profile_description': request.form.get('profile_description'),
            })
        else:
            update_data.update({
                'preferred_language': request.form.get('preferred_language'),
            })

        # Remove None values
        update_data = {k: v for k, v in update_data.items() if v is not None}

        User.update_user(session['user_id'], update_data)
        flash('Profile updated successfully!', 'success')
        return redirect(url_for('main.dashboard', tab='profile'))

    return render_template('update_profile.html', user=user_data)
