from defaults import *  # noqa


SECRET_KEY = open(SECRET_KEY_FILE).read()
DEBUG = False
TEMPLATE_DEBUG = False
ALLOWED_HOSTS = ['.fabexample.us']
STATICFILES_DIRS += ((os.path.join(os.path.abspath(os.sep), 'var', 'www', 'simplex')),)
STATIC_ROOT = os.path.join(os.path.abspath(os.sep), 'usr', 'local', 'simplex', 'static')

CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.memcached.MemcachedCache',
        'LOCATION': '127.0.0.1:11211',
        # We store expensive computations that will be performed infrequently
        # So, we'll just save the results for all computations indefinitely.
        'TIMEOUT': None,
    }
}

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': "[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s",
            'datefmt': "%d/%b/%Y %H:%M:%S",
        }
    },
    'handlers': {
        'file': {
            'level': 'DEBUG',
            'class': 'logging.FileHandler',
            'filename': '/var/log/fabexample.log',
            'formatter': 'verbose',
        }
    },
    'loggers': {
        'django': {
            'handlers': ['file'],
            'propagate': True,
            'level': 'DEBUG',
        },
        'request': {
            'handlers': ['file'],
            'level': 'DEBUG',
        },
    }
}
