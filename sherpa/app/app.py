import logging
import pandas
from flask import Flask
from flask import render_template, flash, redirect

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class SherpaApp(Flask):
    def __init__(self, *args, **kwargs):
        Flask.__init__(self, *args, **kwargs)
        self.parameter_types = {}
    
    def set_results_channel(self, results_channel):
        self.results = pandas.DataFrame()
        self.results_channel = results_channel

    def set_stopping_channel(self, stopping_channel):
        self.stopping_channel = stopping_channel

    def get_results(self):
        # while not self.results_channel.empty():
        #     self.results = self.results_channel.get()
        return self.results_channel.df


app = SherpaApp(__name__)


@app.route('/')
@app.route('/index')
def index():
    """
        Index view.
    """
    results = app.get_results()
    if not results.empty:
        all_ids = set(results['Trial-ID'])
        finished_ids = set(results.loc[results['Status']!='INTERMEDIATE', 'Trial-ID'])
        active_ids = all_ids ^ finished_ids
        active_trials = [{'id': i} for i in active_ids]
        return render_template("index.html",
                               active_trials=active_trials,
                               parameter_types=app.parameter_types,
                               results=[row.to_dict() for idx, row in results.iterrows()])
    else:
        return render_template("index.html",
                               active_trials=[],
                               parameter_types=app.parameter_types,
                               results=[])



@app.route("/stop/<id>", methods=['GET'])
def stop_trial(id):
    """
        Put stopping id on queue.
    """
    app.stopping_channel.put(int(id))
    logger.info("Selected Trial {} to stop.".format(id))
    # logger.debug(app.stopping_channel)
    return redirect('/index')