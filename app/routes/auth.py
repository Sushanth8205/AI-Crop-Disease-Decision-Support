from flask import Blueprint, request, render_template, redirect, url_for, flash, session
from app.models.user import User

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/register', methods=['GET'])
def register():
    return render_template('register.html')

@auth_bp.route('/register/user', methods=['GET', 'POST'])
def register_user():
    if request.method == 'POST':
        email = request.form.get('email')
        
        if User.get_by_email(email):
            flash('Email already registered', 'error')
            return redirect(url_for('auth.register_user'))
            
        data = {
            'name': request.form.get('name'),
            'email': email,
            'password': request.form.get('password'),
            'phone': request.form.get('phone'),
            'location': request.form.get('location'),
            'district': request.form.get('district'),
            'state': request.form.get('state'),
            'lat': request.form.get('lat'),
            'lng': request.form.get('lng'),
            'preferred_language': request.form.get('preferred_language'),
            'role': 'user'
        }
        User.create_user(data)
        flash('User Registration successful! Please login.', 'success')
        return redirect(url_for('auth.login'))
        
    return render_template('register_user.html')

@auth_bp.route('/register/advisor', methods=['GET', 'POST'])
def register_advisor():
    if request.method == 'POST':
        email = request.form.get('email')
        
        if User.get_by_email(email):
            flash('Email already registered', 'error')
            return redirect(url_for('auth.register_advisor'))
            
        data = {
            'name': request.form.get('name'),
            'email': email,
            'password': request.form.get('password'),
            'phone': request.form.get('phone'),
            'location': request.form.get('location'),
            'district': request.form.get('district'),
            'state': request.form.get('state'),
            'lat': request.form.get('lat'),
            'lng': request.form.get('lng'),
            'experience': request.form.get('experience'),
            'qualification': request.form.get('qualification'),
            'crop_specialization': request.form.get('crop_specialization'),
            'disease_specialization': request.form.get('disease_specialization'),
            'available_days': request.form.get('available_days'),
            'available_time_slots': request.form.get('available_time_slots'),
            'profile_description': request.form.get('profile_description'),
            'role': 'advisor'
        }
        User.create_user(data)
        flash('Advisor Registration successful! Please login.', 'success')
        return redirect(url_for('auth.login'))
        
    return render_template('register_advisor.html')

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        user = User.get_by_email(email)
        if user and User.verify_password(password, user['password']):
            session['user_id'] = str(user['_id'])
            session['role'] = user.get('role', 'user')
            session['name'] = user.get('name', '')
            flash(f'Welcome back, {user["name"]}!', 'success')
            return redirect(url_for('main.dashboard'))
            
        flash('Invalid email or password', 'error')
        return redirect(url_for('auth.login'))
        
    return render_template('login.html')

@auth_bp.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'success')
    return redirect(url_for('main.index'))
