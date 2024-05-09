import os
import site
import sys

from werkzeug.middleware.proxy_fix import ProxyFix

if os.getenv("FLASK_ENV", "") != "production":
    print("Python version ", sys.version)
    print("Version info ", sys.version_info)
    print("Python Path ", sys.path)
    print("Python packages path ", site.getsitepackages())
    print("FLASK_ENV: ", os.getenv("FLASK_ENV", ""))


from app import create_app

app = create_app()

# * If Flask is behind a proxy, we need this proxy fix
# * https://flask.palletsprojects.com/en/3.0.x/deploying/proxy_fix/
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

if __name__ == "__main__":
    print("Starting ComplyAi API 🚀")
    print("----------------------->")
    app.run()
