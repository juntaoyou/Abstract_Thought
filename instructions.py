instructions = {
"Expertise": {"A": "Generate a response that can be easily understood by an elementary school student.",
              "B":"Generate a response that only a PhD Student in that specific field could understand."}, 
"Informativeness": {"A":"Generate a response that is concise and to the point, without being verbose.", 
                    "B":"Generate a response that is very informative, without missing any background information."},
"Style":{"A":"Generate a response that is friendly, witty, funny, and humorous, like a close friend.",
         "B":"Generate a response in an unfriendly manner."}
}

doubled_instructions = {
    "Expertise_Informativeness": {
        "AA": "Generate a response that can be easily understood by an elementary school student and that is concise and to the point, without being verbose.", 
        "AB": "Generate a response that can be easily understood by an elementary school student and that is very informative, without missing any background information.",
        "BA": "Generate a response that only a PhD Student in that specific field could understand and that is concise and to the point, without being verbose.",
        "BB": "Generate a response that only a PhD Student in that specific field could understand and that is very informative, without missing any background information."
    },
    "Expertise_Style": {
        "AA": "Generate a response that can be easily understood by an elementary school student and that is friendly, witty, funny, and humorous, like a close friend.", 
        "AB": "Generate a response that can be easily understood by an elementary school student and in an unfriendly manner.",
        "BA": "Generate a response that only a PhD Student in that specific field could understand and that is friendly, witty, funny, and humorous, like a close friend.",
        "BB": "Generate a response that only a PhD Student in that specific field could understand and in an unfriendly manner."
    },
    "Informativeness_Style": {
        "AA": "Generate a response that is concise and to the point, without being verbose and that is friendly, witty, funny, and humorous, like a close friend.", 
        "AB": "Generate a response that is concise and to the point, without being verbose and in an unfriendly manner.",
        "BA": "Generate a response that is very informative, without missing any background information and that is friendly, witty, funny, and humorous, like a close friend.",
        "BB": "Generate a response that is very informative, without missing any background information and in an unfriendly manner."
    },
}
tripled_instructions = {
    "Expertise_Informativeness_Style": {
        "AAA":"Generate a response that can be easily understood by an elementary school student, that is concise and to the point, without being verbose and that is friendly, witty, funny, and humorous, like a close friend.",
        "AAB":"Generate a response that can be easily understood by an elementary school student, that is concise and to the point, without being verbose and in an unfriendly manner.",
        "ABA":"Generate a response that can be easily understood by an elementary school student, that is very informative, without missing any background information and that is friendly, witty, funny, and humorous, like a close friend.",
        "ABB":"Generate a response that can be easily understood by an elementary school student, that is very informative, without missing any background information and in an unfriendly manner.",
        "BAA":"Generate a response that only a PhD Student in that specific field could understand, that is concise and to the point, without being verbose and that is friendly, witty, funny, and humorous, like a close friend.",
        "BAB":"Generate a response that only a PhD Student in that specific field could understand, that is concise and to the point, without being verbose and in an unfriendly manner",
        "BBA":"Generate a response that only a PhD Student in that specific field could understand, that is very informative, without missing any background information and that is friendly, witty, funny, and humorous, like a close friend.",
        "BBB":"Generate a response that only a PhD Student in that specific field could understand, that is very informative, without missing any background information and in an unfriendly manner."
    }
}


correctness_template = (
"You are required to act as a professional scoring model. "
"For the query (user question) and corresponding response (answer content), you must score from a **single dimension** only, without considering the performance in other dimensions."
"A 1-5 point scoring system is adopted, with specific dimension definitions and scoring criteria as follows:\n"
"Dimension: Correctness\n"
"5 (Perfectly correct)- The response is completely correct and accurate to what is requested by the "
"prompt with no necessary details missing and without false, misleading, or hallucinated information. "
"If the prompt asks the assistant to do a task, the task is completely done and addressed in the response"
"(within the limits of the assistant’s capabilities and intended usage).\n"
"4 (Mostly correct)- The response is mostly accurate and correct with a small amount of missing"
"information. It contains no misleading information or hallucinations. If the prompt asks the assistant "
"to perform a task, the task is mostly successfully attempted.\n"
"3 (Partially correct)- The response contains a mix of correct and incorrect information. The "
"response may miss some details, contain misleading information, or minor hallucinations, but is more "
"or less aligned with what the prompt asks for. If the prompt asks the assistant to perform a task, the "
"task is attempted with moderate success but still has clear room for improvement.\n"
"2(Slightly correct)- The response has some correct elements but is mostly wrong or incomplete. "
"The response may contain multiple instances of hallucinated, false and/or misleading information. If "
"the prompt asks the assistant to do a task, the task was attempted with a small amount of success.\n"
"1(Not correct)- The response is completely incorrect. All information provided is wrong, false or "
"hallucinated. If the prompt asks the assistant to do a task, the task is not at all attempted for no good "
"reason, or the wrong task was attempted in the response. The response is completely irrelevant to the "
"prompt.\n"
"Based on the above criteria, score the Helpfulness dimension of the following query and response without explanation:\n"
)

correctness_template = (
"You are required to act as a professional scoring model. "
"For the query (user question) and corresponding response (answer content), you must score from a **single dimension** only, without considering the performance in other dimensions."
"A 1-5 point scoring system is adopted, with specific dimension definitions and scoring criteria as follows:\n"
"Dimension: Correctness\n"
"5 (Perfectly correct)- The response is completely correct and accurate to what is requested by the "
"prompt with no necessary details missing and without false, misleading, or hallucinated information. "
"If the prompt asks the assistant to do a task, the task is completely done and addressed in the response"
"(within the limits of the assistant’s capabilities and intended usage).\n"
"4 (Mostly correct)- The response is mostly accurate and correct with a small amount of missing"
"information. It contains no misleading information or hallucinations. If the prompt asks the assistant "
"to perform a task, the task is mostly successfully attempted.\n"
"3 (Partially correct)- The response contains a mix of correct and incorrect information. The "
"response may miss some details, contain misleading information, or minor hallucinations, but is more "
"or less aligned with what the prompt asks for. If the prompt asks the assistant to perform a task, the "
"task is attempted with moderate success but still has clear room for improvement.\n"
"2(Slightly correct)- The response has some correct elements but is mostly wrong or incomplete. "
"The response may contain multiple instances of hallucinated, false and/or misleading information. If "
"the prompt asks the assistant to do a task, the task was attempted with a small amount of success.\n"
"1(Not correct)- The response is completely incorrect. All information provided is wrong, false or "
"hallucinated. If the prompt asks the assistant to do a task, the task is not at all attempted for no good "
"reason, or the wrong task was attempted in the response. The response is completely irrelevant to the "
"prompt.\n"
"Based on the above criteria, score the Helpfulness dimension of the following query and response without explanation:\n"
)

coherence_template = (
"You are required to act as a professional scoring model. "
"For the query (user question) and corresponding response (answer content), you must score from a **single dimension** only, without considering the performance in other dimensions."
"A 1-5 point scoring system is adopted, with specific dimension definitions and scoring criteria as follows:\n"
"Dimension: Coherence\n"
"5(Perfectly Coherent and Clear)- The response is perfectly clear and self-consistent throughout. "
"There are no contradictory assertions or statements, the writing flows logically and following the "
"train of thought/story is not challenging.\n"
"4(Mostly Coherent and Clear)- The response is mostly clear and coherent, but there may be one or "
"two places where the wording is confusing, the flow of the response is a little hard to follow, or with "
"a small amount of repetitions / irrelevant content. Overall, the response can mostly be followed with "
"a little room for improvement.\n"
"3 (A Little Unclear and/or Incoherent)- The response is a little unclear. There are some incon"
"sistencies or contradictions, run-on sentences, confusing statements, blatant repetitions, significant "
"amounts of irrelevant content, or hard to follow sections of the response.\n"
"2 (Mostly Incoherent and/or Unclear)- The response is mostly hard to follow, with inconsistencies, "
"contradictions, confusing logic flow, unclear language, constant repetitions or mostly irrelevant "
"content used throughout, but there are still some coherent/clear parts.\n"
"1 (Completely Incoherent and/or Unclear)- The response is completely incomprehensible or "
"irrelevant and no clear meaning or sensible message can be discerned from it. The language of the "
"response (spanish) may be inconsistent with prompt (portuguese).\n"
"Based on the above criteria, score the Helpfulness dimension of the following query and response without explanation:\n"
)

expertise_template = (
"You are required to act as a professional scoring model. "
"For the query (user question) and corresponding response (answer content), you must score from a **single dimension** only, without considering the performance in other dimensions."
"A 1-5 point scoring system is adopted, with specific dimension definitions and scoring criteria as follows:\n"
"Dimension: Expertise\n"
"5(Expert)- Deep expertise in the field or area (typically associated with post-graduate education) is \n"
"required to understand the response. It uses specific and technically relevant vocabulary, or elevated "
"language that someone at the simple or basic level may not understand at all. The professional "
"language of a lawyer, scientist, engineer, or doctor falls into this category.\n"
"4 (Advanced)- The response uses a fairly sophisticated vocabulary and terminology. Someone "
"majoring in this subject at a university (post-18 education) would understand the response, while an "
"average adult who does not work or study in this area would not.\n "
"3(Intermediate)- People who have completed up through a high school education (up to age 18)"
"will probably be able to understand the vocabulary and sentence structure used, but those at the basic "
"level or children might struggle to understand the response.\n"
"2(Simple)- The response uses relatively straightforward language and wording, but some schooling "
"through elementary (age 7 to 12) or middle school (age 13- 15) in the language might be required to "
"understand the response.\n"
"1 (Basic)- The response uses very easy to understand language that is clear and completely "
"interpretable by children under 6, adults, and anyone with a functional command of the language.\n"
"Based on the above criteria, score the Helpfulness dimension of the following query and response without explanation:\n"
)

informativeness_template = (
"You are required to act as a professional scoring model. "
"For the query (user question) and corresponding response (answer content), you must score from a **single dimension** only, without considering the performance in other dimensions."
"A 1-5 point scoring system is adopted, with specific dimension definitions and scoring criteria as follows:\n"
"Dimension: Informativeness\n"
"5 (Verbose)- The response is particularly lengthy, wordy, and/or extensive with extra details given "
"what the prompt requested from the assistant model. The response can be verbose regardless of if the "
"length is due to repetition and incoherency or if it is due to rich and insightful detail.\n"
"4 (Moderately Long)- The response is on the longer side but could still have more added to it "
"before it is considered fully detailed or rambling.\n"
"3 (Intermediate Length)- The response isn’t especially long or short given what the prompt is "
"asking of the model. The length is adequate for conveying a full response but isn’t particularly wordy "
"nor particularly concise.\n"
"2 (Pretty Short)- The response is on the shorter side but could still have words, details, and/or text "
"removed before it’s at a bare minimum of what the response is trying to convey.\n"
"1 (Succinct)- The response is short, to the point, and the most concise it can be. No additional "
"information is provided outside of what is requested by the prompt (regardless of if the information "
"or response itself is incorrect, hallucinated, or misleading: a response that gives an incorrect answer "
"can still be succinct).\n"
"Based on the above criteria, score the Helpfulness dimension of the following query and response without explanation:\n"
)

helpfulness_template = (
"You are required to act as a professional scoring model. "
"For the query (user question) and corresponding response (answer content), you must score from a **single dimension** only, without considering the performance in other dimensions."
"A 1-5 point scoring system is adopted, with specific dimension definitions and scoring criteria as follows:\n"
"Dimension: Informativeness\n"
"5(Perfectly helpful)- The response is extremely helpful and completely aligned with the spirit of "
"what the prompt was asking for. It acts on the user’s request accurately, and to the point- without "
"any unnecessary information. If a user request is not possible/inline with desired model behavior, a "
"helpful response provides useful context and rationale even if they do not act on user request directly.\n"
"4 (Mostly helpful)- The response is mostly helpful and mainly aligned with what the user was "
"looking for, but there is still some room for improvement.\n"
"3 (Partially helpful)- The response is partially helpful but misses the overall goal of the user’s "
"query/input in some way. The response did not fully satisfy what the user was looking for.\n"
"2 (Slightly helpful)- The response is borderline helpful and mostly does not capture what the user "
"was looking for, but it is still usable and helpful in a small way.\n"
"1(Not helpful)- The response is not useful or helpful at all. The response completely missed the "
"essence of what the user wanted.\n"
"Based on the above criteria, score the Helpfulness dimension of the following query and response without explanation:\n"
)

user_content = (
"Query: {query}\n"
"response:{response}\n"
"Score:\n"
)



