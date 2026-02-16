from matplotlib import pyplot as plt
from ucimlrepo import fetch_ucirepo

drug_review = fetch_ucirepo(id=461)
# access metadata
print(drug_review.metadata.additional_info.summary, "\n")

# access variable info in tabular format
print(drug_review.variables, "\n")  # Вызывает наши признаки и их подробное описание

# print(drug_review.data.features)  # Так вызываем только признаки
# print(drug_review.data.targets)  # Так вызываем только значения
print(drug_review.data.original, "\n")  # Так вызывает все вместе

reviewID = drug_review.data.original.reviewID
urlDrugName = drug_review.data.original.urlDrugName
rating = drug_review.data.original.rating
effectiveness = drug_review.data.original.effectiveness
sideEffects = drug_review.data.original.sideEffects
condition = drug_review.data.original.condition
benefitsReview = drug_review.data.original.benefitsReview
sideEffectsReview = drug_review.data.original.sideEffectsReview
commentsReview = drug_review.data.original.commentsReview

print(
    "Features:\n",
    reviewID[:1],
    "\n\n",
    urlDrugName[:1],
    "\n\n",
    rating[:1],
    "\n\n",
    effectiveness[:1],
    "\n\n",
    sideEffects[:1],
    "\n\n",
    condition[:1],
    "\n\n",
    benefitsReview[:1],
    "\n\n",
    sideEffectsReview[:1],
    "\n\n",
    commentsReview[:1],
    "\n\n",
)
