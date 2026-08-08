from flask import Flask, render_template, request, jsonify
import csv
import os
from datetime import datetime


app = Flask(__name__)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

EVALUATION_DIR = os.path.join(
    BASE_DIR,
    "evaluation"
)

FEEDBACK_FILE = os.path.join(
    EVALUATION_DIR,
    "manager_feedback.csv"
)

DIGITAL_TWIN_FILE = os.path.join(
    EVALUATION_DIR,
    "digital_twin_results.csv"
)


# ============================================================
# CREATE EVALUATION FOLDER
# ============================================================

os.makedirs(
    EVALUATION_DIR,
    exist_ok=True
)


# ============================================================
# GENERIC CSV READER
# ============================================================

def read_csv_file(filepath):

    if not os.path.exists(filepath):
        return []

    try:

        with open(
            filepath,
            "r",
            newline="",
            encoding="utf-8"
        ) as file:

            reader = csv.DictReader(file)

            return list(reader)

    except Exception as e:

        print("CSV READ ERROR:", e)

        return []


# ============================================================
# HUMAN FEEDBACK
# ============================================================

def read_feedback_records():

    return read_csv_file(
        FEEDBACK_FILE
    )


# ============================================================
# FEEDBACK STATISTICS
# ============================================================

def feedback_statistics():

    records = read_feedback_records()

    accepted = 0
    increased = 0
    reduced = 0

    differences = []

    for record in records:

        decision = str(
            record.get(
                "decision",
                ""
            )
        ).strip().lower()

        if decision == "accepted":
            accepted += 1

        elif decision == "increased":
            increased += 1

        elif decision == "reduced":
            reduced += 1

        try:

            difference = float(
                record.get(
                    "difference",
                    0
                )
            )

            differences.append(
                difference
            )

        except (
            ValueError,
            TypeError
        ):

            pass

    total = (
        accepted +
        increased +
        reduced
    )

    # --------------------------------------------------------
    # AVERAGE ADJUSTMENT
    # --------------------------------------------------------

    if differences:

        average_difference = (
            sum(differences) /
            len(differences)
        )

    else:

        average_difference = 0

    # --------------------------------------------------------
    # ACCEPTANCE RATE
    # --------------------------------------------------------

    if total > 0:

        acceptance_rate = (
            accepted /
            total *
            100
        )

    else:

        acceptance_rate = 0

    # --------------------------------------------------------
    # MANAGER PREFERENCE
    # --------------------------------------------------------

    if average_difference > 5:

        preference = "Usually Increases"

    elif average_difference < -5:

        preference = "Usually Reduces"

    else:

        preference = "Usually Agrees"

    return {

        "acceptance_rate":
            round(
                acceptance_rate,
                2
            ),

        "average_difference":
            round(
                average_difference,
                2
            ),

        "total_decisions":
            total,

        "accepted":
            accepted,

        "increased":
            increased,

        "reduced":
            reduced,

        "preferred_adjustment":
            round(
                average_difference,
                2
            ),

        "preference":
            preference
    }


# ============================================================
# SAVE HUMAN FEEDBACK
# ============================================================

def save_feedback(
    ai_price,
    manager_price,
    decision,
    scenario
):

    os.makedirs(
        EVALUATION_DIR,
        exist_ok=True
    )

    difference = (
        manager_price -
        ai_price
    )

    fieldnames = [

        "ai_price",
        "manager_price",
        "difference",
        "decision",
        "scenario",
        "timestamp"

    ]

    file_exists = os.path.exists(
        FEEDBACK_FILE
    )

    with open(
        FEEDBACK_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        if not file_exists:

            writer.writeheader()

        writer.writerow({

            "ai_price":
                round(
                    ai_price,
                    2
                ),

            "manager_price":
                round(
                    manager_price,
                    2
                ),

            "difference":
                round(
                    difference,
                    2
                ),

            "decision":
                decision,

            "scenario":
                scenario,

            "timestamp":
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
        })


# ============================================================
# DIGITAL TWIN SIMULATION
# ============================================================

def simulate_market(
    event,
    weather,
    competitor_price,
    demand,
    rl_price
):

    competitor_price = float(
        competitor_price
    )

    demand = float(
        demand
    )

    rl_price = float(
        rl_price
    )

    # ========================================================
    # HOTEL CAPACITY
    # ========================================================

    capacity = 100


    # ========================================================
    # WEATHER EFFECT
    # ========================================================

    weather_factor = {

        "Sunny": 1.10,

        "Cloudy": 1.00,

        "Rainy": 0.70

    }.get(
        weather,
        1.00
    )


    # ========================================================
    # EVENT EFFECT
    # ========================================================

    event_factor = {

        "Festival": 1.30,

        "Holiday": 1.20,

        "Conference": 1.25,

        "Normal Day": 1.00

    }.get(
        event,
        1.00
    )


    # ========================================================
    # BASE SIMULATED DEMAND
    # ========================================================

    simulated_demand = (
        demand *
        weather_factor *
        event_factor
    )


    # ========================================================
    # PRICE EFFECT
    #
    # If RL price is much higher than competitor price,
    # demand decreases.
    #
    # If RL price is lower than competitor price,
    # demand increases.
    # ========================================================

    price_ratio = (
        rl_price /
        competitor_price
        if competitor_price > 0
        else 1
    )


    if price_ratio > 1.20:

        price_factor = 0.65

    elif price_ratio > 1.10:

        price_factor = 0.80

    elif price_ratio > 1.00:

        price_factor = 0.90

    elif price_ratio < 0.80:

        price_factor = 1.15

    elif price_ratio < 0.90:

        price_factor = 1.10

    else:

        price_factor = 1.00


    simulated_demand = (
        simulated_demand *
        price_factor
    )


    simulated_demand = max(
        0,
        round(
            simulated_demand
        )
    )


    # ========================================================
    # RL PRICING
    # ========================================================

    rooms_sold = min(
        simulated_demand,
        capacity
    )

    rl_revenue = (
        rooms_sold *
        rl_price
    )


    # ========================================================
    # RULE BASED PRICING
    # ========================================================

    rule_price = competitor_price


    if event in [
        "Festival",
        "Holiday",
        "Conference"
    ]:

        rule_price *= 1.05


    if weather == "Rainy":

        rule_price *= 0.95


    rule_price = round(
        rule_price
    )


    # Rule-based demand

    rule_demand = (
        demand *
        weather_factor
    )


    if event in [
        "Festival",
        "Holiday",
        "Conference"
    ]:

        rule_demand *= event_factor


    rule_rooms = min(
        round(
            rule_demand
        ),
        capacity
    )


    rule_revenue = (
        rule_rooms *
        rule_price
    )


    # ========================================================
    # STATIC PRICING
    # ========================================================

    static_price = 200


    static_demand = (
        demand *
        weather_factor *
        event_factor
    )


    static_rooms = min(
        round(
            static_demand
        ),
        capacity
    )


    static_revenue = (
        static_rooms *
        static_price
    )


    # ========================================================
    # STRATEGY COMPARISON
    # ========================================================

    strategies = {

        "Static Pricing":
            round(
                static_revenue
            ),

        "Rule-Based Pricing":
            round(
                rule_revenue
            ),

        "RL Pricing":
            round(
                rl_revenue
            )
    }


    winner = max(
        strategies,
        key=strategies.get
    )


    # ========================================================
    # OCCUPANCY
    # ========================================================

    occupancy = (
        rooms_sold /
        capacity *
        100
    )


    occupancy = round(
        occupancy,
        2
    )


    # ========================================================
    # RISK ANALYSIS
    #
    # HIGH:
    #   - Very low occupancy
    #   - Rainy + low occupancy
    #   - Very high demand pressure
    #
    # MEDIUM:
    #   - Moderate occupancy
    #   - Some demand/weather risk
    #
    # LOW:
    #   - Healthy occupancy
    # ========================================================

    if (
        occupancy < 25
        or
        (
            weather == "Rainy"
            and
            occupancy < 50
        )
    ):

        risk = "HIGH"


    elif (
        occupancy < 60
        or
        (
            weather == "Rainy"
        )
        or
        (
            price_ratio > 1.20
        )
    ):

        risk = "MEDIUM"


    else:

        risk = "LOW"


    # ========================================================
    # RECOMMENDATION
    # ========================================================

    if occupancy < 25:

        recommendation = (
            "HIGH RISK: Occupancy is very low. "
            "Consider reducing the RL price "
            "to attract more bookings."
        )


    elif occupancy < 60:

        recommendation = (
            "MEDIUM RISK: Occupancy is moderate. "
            "Monitor demand and competitor prices "
            "before increasing the price."
        )


    elif simulated_demand >= 100:

        recommendation = (
            "High demand detected. "
            "The hotel may have an opportunity "
            "to increase the price."
        )


    elif price_ratio > 1.20:

        recommendation = (
            "RL price is significantly higher "
            "than the competitor price. "
            "Consider reducing the price."
        )


    elif weather == "Rainy":

        recommendation = (
            "Rainy weather may reduce bookings. "
            "Consider a competitive price."
        )


    else:

        recommendation = (
            "LOW RISK: Demand and occupancy "
            "conditions are healthy."
        )


    # ========================================================
    # RETURN RESULT
    # ========================================================

    return {

        "event":
            event,

        "weather":
            weather,

        "competitor_price":
            round(
                competitor_price
            ),

        "demand":
            round(
                demand
            ),

        "simulated_demand":
            simulated_demand,

        "rl_price":
            round(
                rl_price
            ),

        "rooms_sold":
            rooms_sold,

        "occupancy":
            occupancy,

        "revenue":
            round(
                rl_revenue
            ),

        "static_revenue":
            round(
                static_revenue
            ),

        "rule_revenue":
            round(
                rule_revenue
            ),

        "winner":
            winner,

        "risk":
            risk,

        "recommendation":
            recommendation,

        "strategies":
            strategies
    }


# ============================================================
# SAVE DIGITAL TWIN RESULT
# ============================================================

def save_simulation(result):

    os.makedirs(
        EVALUATION_DIR,
        exist_ok=True
    )


    fieldnames = [

        "event",
        "weather",
        "competitor_price",
        "demand",
        "simulated_demand",
        "rl_price",
        "rooms_sold",
        "occupancy",
        "revenue",
        "static_revenue",
        "rule_revenue",
        "winner",
        "risk",
        "recommendation",
        "timestamp"

    ]


    file_exists = os.path.exists(
        DIGITAL_TWIN_FILE
    )


    with open(
        DIGITAL_TWIN_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )


        if not file_exists:

            writer.writeheader()


        writer.writerow({

            "event":
                result["event"],

            "weather":
                result["weather"],

            "competitor_price":
                result["competitor_price"],

            "demand":
                result["demand"],

            "simulated_demand":
                result["simulated_demand"],

            "rl_price":
                result["rl_price"],

            "rooms_sold":
                result["rooms_sold"],

            "occupancy":
                result["occupancy"],

            "revenue":
                result["revenue"],

            "static_revenue":
                result["static_revenue"],

            "rule_revenue":
                result["rule_revenue"],

            "winner":
                result["winner"],

            "risk":
                result["risk"],

            "recommendation":
                result["recommendation"],

            "timestamp":
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
        })


# ============================================================
# DASHBOARD
# ============================================================

@app.route("/")
def dashboard():

    feedback = (
        feedback_statistics()
    )


    simulations = read_csv_file(
        DIGITAL_TWIN_FILE
    )


    total_revenue = 0

    occupancies = []

    winners = {}


    for row in simulations:

        try:

            total_revenue += float(
                row.get(
                    "revenue",
                    0
                )
            )


            occupancies.append(
                float(
                    row.get(
                        "occupancy",
                        0
                    )
                )
            )


            winner = row.get(
                "winner",
                ""
            )


            if winner:

                winners[winner] = (
                    winners.get(
                        winner,
                        0
                    ) + 1
                )


        except (
            ValueError,
            TypeError
        ):

            pass


    if occupancies:

        average_occupancy = (
            sum(occupancies) /
            len(occupancies)
        )

    else:

        average_occupancy = 0


    if winners:

        best_strategy = max(
            winners,
            key=winners.get
        )

    else:

        best_strategy = "N/A"


    return render_template(

        "dashboard.html",

        total_revenue=
            round(
                total_revenue,
                2
            ),

        average_occupancy=
            round(
                average_occupancy,
                2
            ),

        best_strategy=
            best_strategy,

        manager_acceptance=
            feedback[
                "acceptance_rate"
            ]
    )


# ============================================================
# HUMAN FEEDBACK PAGE
# ============================================================

@app.route("/human-feedback")
def human_feedback():

    records = (
        read_feedback_records()
    )


    statistics = (
        feedback_statistics()
    )


    return render_template(

        "human_feedback.html",

        records=records,

        statistics=statistics
    )


# ============================================================
# HUMAN FEEDBACK API
# ============================================================

@app.route(
    "/api/feedback",
    methods=["POST"]
)
def add_feedback():

    try:

        data = request.get_json(
            silent=True
        )


        if not data:

            return jsonify({

                "success": False,

                "message":
                    "No feedback data received."

            }), 400


        ai_price = float(
            data.get(
                "ai_price"
            )
        )


        manager_price = float(
            data.get(
                "manager_price"
            )
        )


        scenario = str(
            data.get(
                "scenario",
                "Manual Scenario"
            )
        ).strip()


        if not scenario:

            scenario = "Manual Scenario"


        if ai_price <= 0:

            return jsonify({

                "success": False,

                "message":
                    "AI price must be greater than 0."

            }), 400


        if manager_price <= 0:

            return jsonify({

                "success": False,

                "message":
                    "Manager price must be greater than 0."

            }), 400


        # ----------------------------------------------------
        # DETERMINE DECISION
        # ----------------------------------------------------

        if manager_price == ai_price:

            decision = "Accepted"

        elif manager_price > ai_price:

            decision = "Increased"

        else:

            decision = "Reduced"


        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        save_feedback(

            ai_price,

            manager_price,

            decision,

            scenario

        )


        # ----------------------------------------------------
        # RECALCULATE
        # ----------------------------------------------------

        statistics = (
            feedback_statistics()
        )


        print(
            "--------------------------------"
        )

        print(
            "FEEDBACK SAVED"
        )

        print(
            "AI PRICE:",
            ai_price
        )

        print(
            "MANAGER PRICE:",
            manager_price
        )

        print(
            "DECISION:",
            decision
        )

        print(
            "SCENARIO:",
            scenario
        )

        print(
            "STATISTICS:",
            statistics
        )

        print(
            "--------------------------------"
        )


        return jsonify({

            "success":
                True,

            "decision":
                decision,

            "difference":
                round(
                    manager_price -
                    ai_price,
                    2
                ),

            "statistics":
                statistics

        })


    except (
        ValueError,
        TypeError
    ):

        return jsonify({

            "success": False,

            "message":
                "Please enter valid numeric prices."

        }), 400


    except Exception as e:

        print(
            "FEEDBACK ERROR:",
            e
        )


        return jsonify({

            "success": False,

            "message":
                str(e)

        }), 500


# ============================================================
# DIGITAL TWIN PAGE
# ============================================================

@app.route("/digital-twin")
def digital_twin():

    simulations = read_csv_file(
        DIGITAL_TWIN_FILE
    )


    return render_template(

        "digital_twin.html",

        simulations=simulations
    )


# ============================================================
# DIGITAL TWIN API
# ============================================================

@app.route(
    "/api/simulate",
    methods=["POST"]
)
def api_simulate():

    try:

        data = request.get_json(
            silent=True
        )


        if not data:

            return jsonify({

                "success": False,

                "message":
                    "No simulation data received."

            }), 400


        result = simulate_market(

            data.get(
                "event",
                "Normal Day"
            ),

            data.get(
                "weather",
                "Sunny"
            ),

            data.get(
                "competitor_price"
            ),

            data.get(
                "demand"
            ),

            data.get(
                "rl_price"
            )

        )


        # ----------------------------------------------------
        # SAVE TO CSV
        # ----------------------------------------------------

        save_simulation(
            result
        )


        print(
            "DIGITAL TWIN SAVED:"
        )

        print(
            result
        )


        return jsonify({

            "success":
                True,

            "result":
                result

        })


    except (
        ValueError,
        TypeError,
        KeyError
    ) as e:

        print(
            "SIMULATION ERROR:",
            e
        )


        return jsonify({

            "success": False,

            "message":
                "Invalid simulation input."

        }), 400


    except Exception as e:

        print(
            "DIGITAL TWIN ERROR:",
            e
        )


        return jsonify({

            "success": False,

            "message":
                str(e)

        }), 500


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    app.run(

        debug=True,

        host="127.0.0.1",

        port=5000
    )