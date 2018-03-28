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

class FlamePredict(object):
    @cherrypy.expose
    def index(self):

        # TODO: replace this hardcode list with one generated analysing the 
        # model repoistory
        endpoint = ['CACO2','DIPL1','hERG4']

        # env will setup the jinja2 template rendering
        env = Environment(loader=FileSystemLoader('templates')) 
        tmpl = env.get_template('index.html')

        return tmpl.render(model_list=endpoint)


@cherrypy.expose
class FlamePredictWS(object):

    @cherrypy.tools.accept(media='text/plain')
    def GET(self):
        return 'predict GET'

    # def POST(self, length=8):
    #     some_string = ''.join(random.sample(string.hexdigits, int(length)))
    #     cherrypy.session['mystring'] = some_string
    #     return some_string

    def POST(self, ifile, model, version ):
        predict = Predict(ifile, model, version)

        success, results = predict.run()
        return results

class FlameInfo(object):
    @cherrypy.expose
    def index(self):
        return ''

@cherrypy.expose
class FlameInfoWS(object):

    @cherrypy.tools.accept(media='text/plain')
    def GET(self):
        return 'info GET'

def dir_inform ():
    return "dir_inform"

if __name__ == '__main__':
    conf = {
        '/': {
            'tools.sessions.on': False,
            'tools.staticdir.root': os.path.abspath(os.getcwd())
        },
        '/predictor': {
            'request.dispatch': cherrypy.dispatch.MethodDispatcher(),
            'tools.response_headers.on': True,
            'tools.response_headers.headers': [('Content-Type', 'text/plain')],
        },
        '/static': {
            'tools.staticdir.on': True,
            'tools.staticdir.dir': './public'
        }
    }

    webapp = FlamePredict()
    webapp.predictor = FlamePredictWS()

    infoapp = FlameInfo()
    infoapp.info = FlameInfoWS()
    
    conf2 = {
        '/info': {
            'request.dispatch': cherrypy.dispatch.MethodDispatcher(),
            'tools.response_headers.on': True,
            'tools.response_headers.headers': [('Content-Type', 'text/plain')],
        }
    }

    cherrypy.config.update({'server.socket_host': '0.0.0.0'})

    cherrypy.tree.mount(webapp, '/', conf)
    cherrypy.tree.mount(infoapp, '/info', conf2)

    cherrypy.engine.start()
    cherrypy.engine.block()

    #cherrypy.quickstart(webapp, '/', conf)