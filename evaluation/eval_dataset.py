from typing import List, Dict, Set

EVALUATION_DATASET = [
    {
        "query": "What is the leave policy for employees during probation?",
        "relevant_uids": {
            "data\\Hr Policy\\hr-policy.pdf::page=16::hash=3b4eb6cac433d39c1f44fea584e10092",
            "data\\Hr Policy\\CHEMEXCIL_HR_Policy_2024_08-01-2024.pdf::page=23::hash=4cd5b178a180d165b2d7978bfc33e03c",
        },
    },
    {
        "query": "How does the expense reimbursement approval process work?",
        "relevant_uids": {
            "data\\Finance policy\\Employee-Expense-Reimbursement-Policy-2023-APPROVED-External-Use.pdf::page=3::hash=6b41d416c4548be1840407b94672693e",
            "data\\Finance policy\\Travel-Policy-2017.pdf::page=1::hash=017eee57dbfc8a530ab21e31467023af",
        },
    },
    {
        "query": "What actions are taken if an employee violates IT security policy?",
        "relevant_uids": {
            "data\\It Policy\\Information-Security-Policy.pdf::page=15::hash=a26da9708014a5972480518dfe4d1194",
            "data\\It Policy\\Information-Technology-Cyber-Security-Policy.pdf::page=3::hash=be5539e23eae5b69670e4c2f2f55c3da",  # <-- Space yahan se hata diya hai
        },
    },
    {
        "query": "Is personal device usage allowed on the company network?",
        "relevant_uids": {
            "data\\It Policy\\it-and-cyber-security-policy-1.pdf::page=108::hash=0831e46debc48277b050e1fca7a19faa",
            "data\\It Policy\\IT-and-Cyber-Security-Policy.pdf::page=5::hash=3be5377d212cc6b2c789335f5de5125f",
        },
    },
    {
        "query": "What is the approval limit for employee expense claims?",
        "relevant_uids": {
            "data\\Hr Policy\\CHEMEXCIL_HR_Policy_2024_08-01-2024.pdf::page=49::hash=e97ca65505529b42bf3ccf6df9c1c624",
            "data\\Hr Policy\\CHEMEXCIL_HR_Policy_2024_08-01-2024.pdf::page=9::hash=128ccc405c08ad89c1416ffd0da7fe50",
        },
    },
]