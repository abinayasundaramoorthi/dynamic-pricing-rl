// ============================================================
// CHANGE MANAGER PRICE
// ============================================================

function changeManagerPrice(percent) {

    const aiPrice =
        parseFloat(
            document.getElementById(
                "aiPrice"
            ).value
        );

    const managerInput =
        document.getElementById(
            "managerPrice"
        );

    if (isNaN(aiPrice)) {

        showMessage(
            "Please enter a valid AI price.",
            "error"
        );

        return;
    }

    const newPrice =
        aiPrice *
        (1 + percent / 100);

    managerInput.value =
        Math.round(newPrice);
}


// ============================================================
// SUBMIT FEEDBACK
// ============================================================

async function submitFeedback(type) {

    const aiPrice =
        parseFloat(
            document.getElementById(
                "aiPrice"
            ).value
        );

    const managerInput =
        document.getElementById(
            "managerPrice"
        );

    const scenario =
        document.getElementById(
            "scenario"
        ).value.trim()
        || "Manual Scenario";


    if (isNaN(aiPrice)) {

        showMessage(
            "Please enter a valid AI price.",
            "error"
        );

        return;
    }


    let managerPrice;


    // --------------------------------------------------------
    // ACCEPT
    // --------------------------------------------------------

    if (type === "accept") {

        managerPrice =
            aiPrice;

        managerInput.value =
            aiPrice;
    }


    // --------------------------------------------------------
    // INCREASE
    // --------------------------------------------------------

    else if (type === "increase") {

        managerPrice =
            Math.round(
                aiPrice * 1.05
            );

        managerInput.value =
            managerPrice;
    }


    // --------------------------------------------------------
    // REDUCE
    // --------------------------------------------------------

    else if (type === "reduce") {

        managerPrice =
            Math.round(
                aiPrice * 0.95
            );

        managerInput.value =
            managerPrice;
    }


    // --------------------------------------------------------
    // CUSTOM
    // --------------------------------------------------------

    else if (type === "custom") {

        managerPrice =
            parseFloat(
                managerInput.value
            );
    }


    if (
        isNaN(managerPrice) ||
        managerPrice <= 0
    ) {

        showMessage(
            "Please enter a valid manager price.",
            "error"
        );

        return;
    }


    // --------------------------------------------------------
    // SEND TO FLASK
    // --------------------------------------------------------

    try {

        const response =
            await fetch(
                "/api/feedback",
                {

                    method: "POST",

                    headers: {

                        "Content-Type":
                            "application/json"

                    },

                    body:
                        JSON.stringify({

                            ai_price:
                                aiPrice,

                            manager_price:
                                managerPrice,

                            scenario:
                                scenario

                        })

                }
            );


        const data =
            await response.json();


        if (!response.ok ||
            !data.success) {

            showMessage(

                data.message ||
                "Unable to save feedback.",

                "error"

            );

            return;
        }


        // ----------------------------------------------------
        // UPDATE STATISTICS
        // ----------------------------------------------------

        updateStatistics(
            data.statistics
        );


        // ----------------------------------------------------
        // ADD ROW
        // ----------------------------------------------------

        addHistoryRow(

            aiPrice,

            managerPrice,

            data.decision,

            scenario

        );


        showMessage(

            "✓ Feedback saved successfully!",

            "success"

        );


    }

    catch (error) {

        console.error(
            "Feedback error:",
            error
        );

        showMessage(

            "Server error. Please check Flask.",

            "error"

        );
    }
}


// ============================================================
// UPDATE STATISTICS
// ============================================================

function updateStatistics(stats) {

    document.getElementById(
        "acceptanceRate"
    ).innerText =
        stats.acceptance_rate + "%";


    document.getElementById(
        "averageDifference"
    ).innerText =
        "₹" +
        stats.average_difference;


    document.getElementById(
        "totalDecisions"
    ).innerText =
        stats.total_decisions;


    document.getElementById(
        "accepted"
    ).innerText =
        stats.accepted;


    document.getElementById(
        "increased"
    ).innerText =
        stats.increased;


    document.getElementById(
        "reduced"
    ).innerText =
        stats.reduced;
}


// ============================================================
// ADD HISTORY ROW
// ============================================================

function addHistoryRow(
    aiPrice,
    managerPrice,
    decision,
    scenario
) {

    const table =
        document.getElementById(
            "feedbackTable"
        );


    if (!table) {

        console.error(
            "feedbackTable not found"
        );

        return;
    }


    const difference =
        managerPrice -
        aiPrice;


    const row =
        document.createElement(
            "tr"
        );


    const timestamp =
        new Date().toLocaleString();


    row.innerHTML = `

        <td>
            ₹${aiPrice}
        </td>

        <td>
            ₹${managerPrice}
        </td>

        <td>
            ₹${difference}
        </td>

        <td>
            <span class="badge">
                ${decision}
            </span>
        </td>

        <td>
            ${scenario}
        </td>

        <td>
            ${timestamp}
        </td>

    `;


    table.prepend(
        row
    );
}


// ============================================================
// MESSAGE
// ============================================================

function showMessage(
    message,
    type
) {

    const element =
        document.getElementById(
            "feedbackMessage"
        );


    if (!element) {

        alert(message);

        return;
    }


    element.innerText =
        message;


    element.className =
        "message " + type;
}