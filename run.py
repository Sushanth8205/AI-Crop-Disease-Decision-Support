import os
from app import create_app

app = create_app()

if __name__ == '__main__':
    # Set use_reloader=False to prevent debugpy / IDE debugger from halting on SystemExit: 3
    # and to prevent heavy PyTorch ML models from loading into RAM twice.
    use_reloader = os.getenv('FLASK_USE_RELOADER', 'false').lower() in ('true', '1')
    port = int(os.environ.get('PORT', 8030))
    app.run(debug=True, use_reloader=use_reloader, host='0.0.0.0', port=port)
