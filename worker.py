
from app import create_app
from workers import wsgi

app = create_app()

Default = wsgi.entrypoint(app)
