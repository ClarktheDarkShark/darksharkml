# models.py

from db import db

class DailyStats(db.Model):
    __tablename__ = 'daily_stats'

    # Date: date of the stream (YYYY-MM-DD)
    stream_date = db.Column(
        db.Date,
        primary_key=True
    )  # e.g. date(2025, 6, 8)

    # String: day of week ("Monday" – "Sunday")
    day_of_week = db.Column(
        db.String(9),
        nullable=False
    )  # e.g. "Sunday"

    # Boolean: whether it was a weekend
    is_weekend = db.Column(
        db.Boolean,
        nullable=False
    )  # e.g. True

    # Boolean: whether it was a public/major holiday
    is_holiday = db.Column(
        db.Boolean,
        nullable=False
    )  # e.g. False

    # Time: when the stream started
    stream_start_time = db.Column(
        db.Time,
        nullable=False,
        primary_key=True,
    )  # e.g. time(19, 30)

    # Integer: days since previous stream
    days_since_previous_stream = db.Column(
        db.Integer,
        nullable=False
    )  # e.g. 2

    # Integer: stream duration in minutes
    stream_duration = db.Column(
        db.Integer,
        nullable=False
    )  # e.g. 180

    # Float: average concurrent viewers
    avg_concurrent_viewers = db.Column(
        db.Float,
        nullable=False
    )  # e.g. 45.2

    # Integer: peak concurrent viewers
    peak_concurrent_viewers = db.Column(
        db.Integer,
        nullable=False
    )  # e.g. 120

    # Integer: total unique viewers
    unique_viewers = db.Column(
        db.Integer,
        nullable=False
    )  # e.g. 200

    # Float: viewer growth rate ((end − start) / start)
    viewer_growth_rate = db.Column(
        db.Float,
        nullable=False
    )  # e.g. 0.25

    # Integer: total number of chat messages
    total_num_chats = db.Column(
        db.Integer,
        nullable=False
    )  # e.g. 3075

    # Integer: total number of distinct chatters
    total_chatters = db.Column(
        db.Integer,
        nullable=False
    )  # e.g. 125

    # Float: chat messages per minute
    chat_msgs_per_minute = db.Column(
        db.Float,
        nullable=False
    )  # e.g. 17.1

    # Integer: total emotes used
    total_emotes_used = db.Column(
        db.Integer,
        nullable=False
    )  # e.g. 350

    # Integer: unique emotes used
    unique_emotes_used = db.Column(
        db.Integer,
        nullable=False
    )  # e.g. 42

    # Integer: followers at start of stream
    followers_start = db.Column(
        db.Integer,
        nullable=False
    )  # e.g. 1500

    # Integer: followers at end of stream
    followers_end = db.Column(
        db.Integer,
        nullable=False
    )  # e.g. 1525

    # Integer: net follower change
    net_follower_change = db.Column(
        db.Integer,
        nullable=False
    )  # e.g. 20

    # Integer: total subscriptions gained (target)
    total_subscriptions = db.Column(
        db.Integer,
        nullable=False
    )  # e.g. 10

    # Integer: new Tier 1 subscriptions
    new_subscriptions_t1 = db.Column(
        db.Integer,
        nullable=False
    )  # e.g. 8

    # Integer: new Tier 2 & 3 subscriptions
    new_subscriptions_t2_t3 = db.Column(
        db.Integer,
        nullable=False
    )  # e.g. 2

    # Integer: resubscriptions (renewals)
    resubscriptions = db.Column(
        db.Integer,
        nullable=False
    )  # e.g. 3

    # Integer: gifted subs received
    gifted_subs_received = db.Column(
        db.Integer,
        nullable=False
    )  # e.g. 5

    # Integer: gifted subs given
    gifted_subs_given = db.Column(
        db.Integer,
        nullable=False
    )  # e.g. 1

    # Integer: subscription cancellations
    subscription_cancellations = db.Column(
        db.Integer,
        nullable=False
    )  # e.g. 1

    # Integer: total bits donated
    bits_donated = db.Column(
        db.Integer,
        nullable=False
    )  # e.g. 1000

    # Integer: number of donation events
    donation_events_count = db.Column(
        db.Integer,
        nullable=False
    )  # e.g. 12

    # Float: total donation amount (USD)
    total_donation_amount = db.Column(
        db.Float,
        nullable=False
    )  # e.g. 45.50

    # Integer: raids received
    raids_received = db.Column(
        db.Integer,
        nullable=False
    )  # e.g. 2

    # Integer: raid viewers received
    raid_viewers_received = db.Column(
        db.Integer,
        nullable=False
    )  # e.g. 150

    # Integer: polls run
    polls_run = db.Column(
        db.Integer,
        nullable=False
    )  # e.g. 3

    # Integer: total poll participation (votes)
    poll_participation = db.Column(
        db.Integer,
        nullable=False
    )  # e.g. 45

    # Integer: predictions run
    predictions_run = db.Column(
        db.Integer,
        nullable=False
    )  # e.g. 2

    # Integer: prediction participants
    prediction_participants = db.Column(
        db.Integer,
        nullable=False
    )  # e.g. 30

    # String: game or category played
    game_category = db.Column(
        db.String(128),
        nullable=False
    )  # e.g. "Just Chatting"

    # Integer: category changes count
    category_changes = db.Column(
        db.Integer,
        nullable=False
    )  # e.g. 1

    # Integer: stream title length (characters)
    title_length = db.Column(
        db.Integer,
        nullable=False
    )  # e.g. 42

    # Boolean: title included "giveaway"
    has_giveaway = db.Column(
        db.Boolean,
        nullable=False
    )  # e.g. True

    # Boolean: title included "Q&A"
    has_qna = db.Column(
        db.Boolean,
        nullable=False
    )  # e.g. False

    # JSON: tags applied to the stream
    tags = db.Column(
        db.JSON,
        nullable=True
    )  # e.g. ["AI", "Gaming", "Chat"]

    # Integer: total moderation actions
    moderation_actions = db.Column(
        db.Integer,
        nullable=False
    )  # e.g. 5

    # Integer: messages deleted by moderators
    messages_deleted = db.Column(
        db.Integer,
        nullable=False
    )  # e.g. 2

    # Integer: timeouts or bans issued
    timeouts_bans = db.Column(
        db.Integer,
        nullable=False
    )  # e.g. 1

    # Float: average chat sentiment score
    avg_sentiment_score = db.Column(
        db.Float,
        nullable=True
    )  # e.g. 0.75

    # Float: positive-to-negative message ratio
    positive_negative_ratio = db.Column(
        db.Float,
        nullable=True
    )  # e.g. 3.5

    # Float: subscriptions per average viewer
    subs_per_avg_viewer = db.Column(
        db.Float,
        nullable=False
    )  # e.g. 0.22

    # Float: chat messages per viewer
    chat_msgs_per_viewer = db.Column(
        db.Float,
        nullable=False
    )  # e.g. 1.5

    # Float: 7-day moving average of subscriptions
    subs_7d_moving_avg = db.Column(
        db.Float,
        nullable=True
    )  # e.g. 12.4

    # Float: 7-day moving average of subscriptions
    subs_3d_moving_avg = db.Column(
        db.Float,
        nullable=True
    )  # e.g. 12.4

    # Float: 3-day moving average of average viewers
    viewers_3d_moving_avg = db.Column(
        db.Float,
        nullable=True
    )  # e.g. 52.1

    # Float: day-over-day change in peak viewers
    day_over_day_peak_change = db.Column(
        db.Float,
        nullable=True
    )  # e.g. -10.0

    # DateTime: record creation timestamp
    created_at = db.Column(
        db.DateTime,
        server_default=db.func.now()
    ) 

    gift_subs_bool = db.Column(
        db.Boolean,
        nullable=False
    ) 

    stream_name = db.Column(
        db.String(128),
        nullable=False
    )

    def __repr__(self):
        return f"<DailyStats date={self.stream_date!r}>"



class TimeSeries(db.Model):
    __tablename__ = 'live_stream'

    id             = db.Column(db.Integer, primary_key=True, autoincrement=True)
    stream_name    = db.Column(db.String(128), nullable=False, index=True)
    snapshot_time  = db.Column(db.DateTime, nullable=False, index=True, server_default=db.func.now())

    # Date: date of the stream (YYYY-MM-DD)
    stream_date = db.Column(
        db.Date
    )  # e.g. date(2025, 6, 8)

    # String: day of week ("Monday" – "Sunday")
    day_of_week = db.Column(
        db.String(9),
        nullable=False
    )  # e.g. "Sunday"

    # Boolean: whether it was a weekend
    is_weekend = db.Column(
        db.Boolean,
        nullable=False
    )  # e.g. True

    # Boolean: whether it was a public/major holiday
    is_holiday = db.Column(
        db.Boolean,
        nullable=False
    ) 

    stream_start_time = db.Column(
        db.Time,
        nullable=False
    ) 

    # Integer: days since previous stream
    days_since_previous_stream = db.Column(
        db.Integer,
        nullable=False
    )  # e.g. 2

    # Integer: stream duration in minutes
    stream_duration = db.Column(
        db.Integer,
        nullable=False
    )  # e.g. 180

    # Float: average concurrent viewers
    avg_concurrent_viewers = db.Column(
        db.Float,
        nullable=False
    )  # e.g. 45.2

    # Integer: peak concurrent viewers
    peak_concurrent_viewers = db.Column(
        db.Integer,
        nullable=False
    )  # e.g. 120

    # Integer: total unique viewers
    unique_viewers = db.Column(
        db.Integer,
        nullable=False
    )  # e.g. 200

    # Float: viewer growth rate ((end − start) / start)
    viewer_growth_rate = db.Column(
        db.Float,
        nullable=False
    )  # e.g. 0.25

    # Integer: total number of chat messages
    total_num_chats = db.Column(
        db.Integer,
        nullable=False
    )  # e.g. 3075

    # Integer: total number of distinct chatters
    total_chatters = db.Column(
        db.Integer,
        nullable=False
    )  # e.g. 125

    # Float: chat messages per minute
    chat_msgs_per_minute = db.Column(
        db.Float,
        nullable=False
    )  # e.g. 17.1

    # Integer: total emotes used
    total_emotes_used = db.Column(
        db.Integer,
        nullable=False
    )  # e.g. 350

    # Integer: unique emotes used
    unique_emotes_used = db.Column(
        db.Integer,
        nullable=False
    )  # e.g. 42

    # Integer: followers at start of stream
    followers_start = db.Column(
        db.Integer,
        nullable=False
    )  # e.g. 1500

    # Integer: followers at end of stream
    followers_end = db.Column(
        db.Integer,
        nullable=False
    )  # e.g. 1525

    # Integer: net follower change
    net_follower_change = db.Column(
        db.Integer,
        nullable=False
    )  # e.g. 20

    # Integer: total subscriptions gained (target)
    total_subscriptions = db.Column(
        db.Integer,
        nullable=False
    )  # e.g. 10

    # Integer: new Tier 1 subscriptions
    new_subscriptions_t1 = db.Column(
        db.Integer,
        nullable=False
    )  # e.g. 8

    # Integer: new Tier 2 & 3 subscriptions
    new_subscriptions_t2_t3 = db.Column(
        db.Integer,
        nullable=False
    )  # e.g. 2

    # Integer: resubscriptions (renewals)
    resubscriptions = db.Column(
        db.Integer,
        nullable=False
    )  # e.g. 3

    # Integer: gifted subs received
    gifted_subs_received = db.Column(
        db.Integer,
        nullable=False
    )  # e.g. 5

    # Integer: gifted subs given
    gifted_subs_given = db.Column(
        db.Integer,
        nullable=False
    )  # e.g. 1

    # Integer: subscription cancellations
    subscription_cancellations = db.Column(
        db.Integer,
        nullable=False
    )  # e.g. 1

    # Integer: total bits donated
    bits_donated = db.Column(
        db.Integer,
        nullable=False
    )  # e.g. 1000

    # Integer: number of donation events
    donation_events_count = db.Column(
        db.Integer,
        nullable=False
    )  # e.g. 12

    # Float: total donation amount (USD)
    total_donation_amount = db.Column(
        db.Float,
        nullable=False
    )  # e.g. 45.50

    # Integer: raids received
    raids_received = db.Column(
        db.Integer,
        nullable=False
    )  # e.g. 2

    # Integer: raid viewers received
    raid_viewers_received = db.Column(
        db.Integer,
        nullable=False
    )  # e.g. 150

    # Integer: polls run
    polls_run = db.Column(
        db.Integer,
        nullable=False
    )  # e.g. 3

    # Integer: total poll participation (votes)
    poll_participation = db.Column(
        db.Integer,
        nullable=False
    )  # e.g. 45

    # Integer: predictions run
    predictions_run = db.Column(
        db.Integer,
        nullable=False
    )  # e.g. 2

    # Integer: prediction participants
    prediction_participants = db.Column(
        db.Integer,
        nullable=False
    )  # e.g. 30

    # String: game or category played
    game_category = db.Column(
        db.String(128),
        nullable=False
    )  # e.g. "Just Chatting"

    # Integer: category changes count
    category_changes = db.Column(
        db.Integer,
        nullable=False
    )  # e.g. 1

    # Integer: stream title length (characters)
    title_length = db.Column(
        db.Integer,
        nullable=False
    )  # e.g. 42

    # Boolean: title included "giveaway"
    has_giveaway = db.Column(
        db.Boolean,
        nullable=False
    )  # e.g. True

    # Boolean: title included "Q&A"
    has_qna = db.Column(
        db.Boolean,
        nullable=False
    )  # e.g. False

    # JSON: tags applied to the stream
    tags = db.Column(
        db.JSON,
        nullable=True
    )  # e.g. ["AI", "Gaming", "Chat"]

    # Integer: total moderation actions
    moderation_actions = db.Column(
        db.Integer,
        nullable=False
    )  # e.g. 5

    # Integer: messages deleted by moderators
    messages_deleted = db.Column(
        db.Integer,
        nullable=False
    )  # e.g. 2

    # Integer: timeouts or bans issued
    timeouts_bans = db.Column(
        db.Integer,
        nullable=False
    )  # e.g. 1

    # Float: average chat sentiment score
    avg_sentiment_score = db.Column(
        db.Float,
        nullable=True
    )  # e.g. 0.75

    # Float: positive-to-negative message ratio
    positive_negative_ratio = db.Column(
        db.Float,
        nullable=True
    )  # e.g. 3.5

    # Float: subscriptions per average viewer
    subs_per_avg_viewer = db.Column(
        db.Float,
        nullable=False
    )  # e.g. 0.22

    # Float: chat messages per viewer
    chat_msgs_per_viewer = db.Column(
        db.Float,
        nullable=False
    )  # e.g. 1.5

    # Float: 7-day moving average of subscriptions
    subs_7d_moving_avg = db.Column(
        db.Float,
        nullable=True
    )  # e.g. 12.4

    # Float: 7-day moving average of subscriptions
    subs_3d_moving_avg = db.Column(
        db.Float,
        nullable=True
    )  # e.g. 12.4

    # Float: 3-day moving average of average viewers
    viewers_3d_moving_avg = db.Column(
        db.Float,
        nullable=True
    )  # e.g. 52.1

    # Float: day-over-day change in peak viewers
    day_over_day_peak_change = db.Column(
        db.Float,
        nullable=True
    )  # e.g. -10.0

    # DateTime: record creation timestamp
    created_at = db.Column(
        db.DateTime,
        server_default=db.func.now()
    ) 

    gift_subs_bool = db.Column(
        db.Boolean,
        nullable=False
    ) 

    def __repr__(self):
        return f"<TimeSeries date={self.stream_date!r}>"
