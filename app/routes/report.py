import os
from flask import Blueprint, send_file, current_app, flash, redirect, url_for, session
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from app.models.detection import Detection
from app.models.user import User
import time

report_bp = Blueprint('report', __name__)

@report_bp.route('/generate/<detection_id>')
def generate_report(detection_id):
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))
        
    detection = Detection.get_detection(detection_id)
    if not detection or str(detection.get('user_id')) != session['user_id']:
        flash('Unauthorized access to report', 'error')
        return redirect(url_for('main.dashboard'))
        
    user = User.get_by_id(session['user_id'])
    
    # Create PDF filename
    pdf_filename = f"report_{detection_id}.pdf"
    pdf_path = os.path.join(current_app.config['UPLOAD_FOLDER'], pdf_filename)
    
    # Build PDF
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    story.append(Paragraph("CropCare AI - Diagnosis Report", styles['Title']))
    story.append(Spacer(1, 12))
    
    # User Details
    story.append(Paragraph(f"<b>Farmer Name:</b> {user.get('name', 'N/A')}", styles['Normal']))
    story.append(Paragraph(f"<b>Location:</b> {user.get('location', 'N/A')}, {user.get('district', 'N/A')}, {user.get('state', 'N/A')}", styles['Normal']))
    story.append(Paragraph(f"<b>Date:</b> {detection.get('timestamp').strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
    story.append(Spacer(1, 12))
    
    # Detection Details
    story.append(Paragraph("<b>Detection Results</b>", styles['Heading2']))
    story.append(Paragraph(f"<b>Crop:</b> {detection.get('crop')}", styles['Normal']))
    story.append(Paragraph(f"<b>Disease:</b> {detection.get('disease')}", styles['Normal']))
    story.append(Paragraph(f"<b>Confidence Score:</b> {detection.get('confidence')}%", styles['Normal']))
    story.append(Spacer(1, 12))
    
    # Image
    image_full_path = os.path.join(current_app.root_path, 'static', detection.get('image_path'))
    if os.path.exists(image_full_path):
        img = RLImage(image_full_path, width=300, height=300)
        story.append(img)
        story.append(Spacer(1, 12))
        
    # Recommendations (Mocked text, normally pulled from Knowledge Base)
    story.append(Paragraph("<b>Recommendations</b>", styles['Heading2']))
    if detection.get('disease').lower() != 'healthy':
        story.append(Paragraph("<b>Symptoms:</b> Leaf spots, discoloration, and wilting.", styles['Normal']))
        story.append(Paragraph("<b>Precautions:</b> Remove infected leaves immediately. Ensure proper spacing between plants.", styles['Normal']))
        story.append(Paragraph("<b>Management:</b> Apply recommended fungicide. Consult an Agricultural Advisor for severe cases.", styles['Normal']))
    else:
        story.append(Paragraph("The crop appears healthy. Continue normal care and monitoring.", styles['Normal']))
        
    doc.build(story)
    
    return send_file(pdf_path, as_attachment=True, download_name=f"Diagnosis_{detection.get('crop')}.pdf")
