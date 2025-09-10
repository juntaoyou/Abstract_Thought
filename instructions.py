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