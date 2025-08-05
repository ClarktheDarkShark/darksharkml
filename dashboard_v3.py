# app/routes/dashboard_v3.py
# --------------------------------------------------------------------------
from flask import Blueprint, jsonify, request
from services.recommendation_service import get_stream_recommendations

dash_v3 = Blueprint("dash_v3", __name__, url_prefix="")

@dash_v3.route("/v3", methods=["GET"])
def api_recommendations():
    """
    Return cached recommendations for a given streamer as JSON.

    Query string:
        ?stream=<streamer_name>   (optional)

    If no stream is supplied or the name is unknown, the service falls back
    to the most-frequent streamer in the inference dataset.  Because the heavy
    work is done by the refresher thread, this call returns in milliseconds.
    """
    stream_name = request.args.get("stream")
    data = get_stream_recommendations(stream_name)
    return jsonify(data)
