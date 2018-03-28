#! -*- coding: utf-8 -*-

##    Description    Flame web-service
##
##    Authors:       Manuel Pastor (manuel.pastor@upf.edu)
##
##    Copyright 2018 Manuel Pastor
##
##    This file is part of Flame
##
##    Flame is free software: you can redistribute it and/or modify
##    it under the terms of the GNU General Public License as published by
##    the Free Software Foundation version 3.
##
##    Flame is distributed in the hope that it will be useful,
##    but WITHOUT ANY WARRANTY; without even the implied warranty of
##    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##    GNU General Public License for more details.
##
##    You should have received a copy of the GNU General Public License
##    along with Flame. If not, see <http://www.gnu.org/licenses/>.

import os
import cherrypy
from jinja2 import Environment 
from jinja2 import FileSystemLoader

from predict import Predict
import util.utils as utils

class FlamePredict(object):
    @cherrypy.expose
    def index(self):

        # analysing the model repoistory
        rdir = utils.root_path()
        endpoint = [x for x in os.listdir (rdir)]

        # env will setup the jinja2 template rendering
        env = Environment(loader=FileSystemLoader('templates')) 
        tmpl = env.get_template('index.html')

        return tmpl.render(model_list=endpoint)


@cherrypy.expose
class FlamePredictWS(object):

    @cherrypy.tools.accept(media='text/plain')

    def POST(self, ifile, model, version ):

        #TODO: check if changing models manages child classes correctly
        try:
            predict = Predict(ifile, model, version)
            success, results = predict.run()
        except:
            raise cherrypy.HTTPError(500)

        return results

@cherrypy.expose
class FlameInfoWS(object):

    @cherrypy.tools.accept(media='text/plain')
    def GET(self):
        return 'GET info'

@cherrypy.expose
class FlameDirWS(object):

    @cherrypy.tools.accept(media='text/plain')
    def GET(self):
        return 'GET dir'

if __name__ == '__main__':
    conf = {
        '/': {
            'tools.sessions.on': False,
            'tools.staticdir.root': os.path.abspath(os.getcwd())
        },
        # '/favicon.ico': {
        #     'tools.staticfile.on': True,
        #     'tools.staticfile.filename': '/static/images/etransafe.ico'
        # },
        '/info': {
            'request.dispatch': cherrypy.dispatch.MethodDispatcher(),
            'tools.response_headers.on': True,
            'tools.response_headers.headers': [('Content-Type', 'text/plain')]
        },
        '/dir': {
            'request.dispatch': cherrypy.dispatch.MethodDispatcher(),
            'tools.response_headers.on': True,
            'tools.response_headers.headers': [('Content-Type', 'text/plain')]
        },
        '/predict': {
            'request.dispatch': cherrypy.dispatch.MethodDispatcher(),
            'tools.response_headers.on': True,
            'tools.response_headers.headers': [('Content-Type', 'text/plain')]
        },
        '/static': {
            'tools.staticdir.on': True,
            'tools.staticdir.dir': './public'
        }
    }
    webapp = FlamePredict()
    webapp.info = FlameInfoWS()
    webapp.dir = FlameDirWS()
    webapp.predict = FlamePredictWS()
    
    cherrypy.config.update({'server.socket_host': '0.0.0.0'})

    cherrypy.quickstart(webapp, '/', conf)
