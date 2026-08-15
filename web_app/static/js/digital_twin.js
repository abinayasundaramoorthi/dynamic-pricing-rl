// ============================================================
// RUN DIGITAL TWIN
// ============================================================

async function runSimulation() {

    const event =
        document.getElementById(
            "event"
        ).value;


    const weather =
        document.getElementById(
            "weather"
        ).value;


    const competitorPrice =
        parseFloat(
            document.getElementById(
                "competitorPrice"
            ).value
        );


    const demand =
        parseFloat(
            document.getElementById(
                "demand"
            ).value
        );


    const rlPrice =
        parseFloat(
            document.getElementById(
                "rlPrice"
            ).value
        );


    if (
        isNaN(competitorPrice) ||
        isNaN(demand) ||
        isNaN(rlPrice)
    ) {

        alert(
            "Please enter valid values."
        );

        return;
    }


    try {

        const response =
            await fetch(
                "/api/simulate",
                {

                    method: "POST",

                    headers: {

                        "Content-Type":
                            "application/json"

                    },

                    body:
                        JSON.stringify({

                            event:
                                event,

                            weather:
                                weather,

                            competitor_price:
                                competitorPrice,

                            demand:
                                demand,

                            rl_price:
                                rlPrice

                        })

                }
            );


        const data =
            await response.json();


        if (!response.ok ||
            !data.success) {

            alert(
                data.message ||
                "Simulation failed."
            );

            return;
        }


        displaySimulation(
            data.result
        );

    }

    catch (error) {

        console.error(
            "Simulation error:",
            error
        );

        alert(
            "Server error. Check Flask console."
        );
    }
}


// ============================================================
// DISPLAY RESULT
// ============================================================

function displaySimulation(
    result
) {

    const resultBox =
        document.getElementById(
            "simulationResult"
        );


    if (resultBox) {

        resultBox.classList.remove(
            "hidden"
        );
    }


    const resultDemand =
        document.getElementById(
            "resultDemand"
        );

    if (resultDemand) {

        resultDemand.innerText =
            result.simulated_demand;
    }


    const resultRooms =
        document.getElementById(
            "resultRooms"
        );

    if (resultRooms) {

        resultRooms.innerText =
            result.rooms_sold;
    }


    const resultRevenue =
        document.getElementById(
            "resultRevenue"
        );

    if (resultRevenue) {

        resultRevenue.innerText =
            "₹" +
            result.revenue;
    }


    const resultWinner =
        document.getElementById(
            "resultWinner"
        );

    if (resultWinner) {

        resultWinner.innerText =
            result.winner;
    }


    const resultRisk =
        document.getElementById(
            "resultRisk"
        );

    if (resultRisk) {

        resultRisk.innerText =
            result.risk;
    }


    const recommendation =
        document.getElementById(
            "recommendation"
        );

    if (recommendation) {

        recommendation.innerText =
            result.recommendation;
    }


    displayStrategies(
        result.strategies
    );


    addSimulationRow(
        result
    );
}


// ============================================================
// STRATEGY COMPARISON
// ============================================================

function displayStrategies(
    strategies
) {

    const container =
        document.getElementById(
            "strategyComparison"
        );


    if (!container) {
        return;
    }


    container.innerHTML = "";


    const sorted =
        Object.entries(
            strategies
        ).sort(
            (a, b) =>
                b[1] - a[1]
        );


    sorted.forEach(
        ([name, revenue], index) => {

            const div =
                document.createElement(
                    "div"
                );


            div.className =
                "strategy-row";


            const trophy =
                index === 0
                    ? "🏆 "
                    : "";


            div.innerHTML = `

                <span>
                    ${trophy}${name}
                </span>

                <strong>
                    ₹${revenue}
                </strong>

            `;


            container.appendChild(
                div
            );
        }
    );
}


// ============================================================
// ADD SIMULATION HISTORY ROW
// ============================================================

function addSimulationRow(
    result
) {

    const table =
        document.getElementById(
            "simulationTable"
        );


    if (!table) {

        console.error(
            "simulationTable not found"
        );

        return;
    }


    const row =
        document.createElement(
            "tr"
        );


    row.innerHTML = `

        <td>
            ${result.event}
        </td>

        <td>
            ${result.weather}
        </td>

        <td>
            ₹${result.competitor_price}
        </td>

        <td>
            ${result.demand}
        </td>

        <td>
            ₹${result.rl_price}
        </td>

        <td>
            ${result.rooms_sold}
        </td>

        <td>
            ₹${result.revenue}
        </td>

        <td>
            <span class="badge">
                ${result.winner}
            </span>
        </td>

        <td>
            <span class="risk">
                ${result.risk}
            </span>
        </td>

    `;


    table.prepend(
        row
    );
}