import math
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from enum import StrEnum

from pydantic import BaseModel, Field

from agent import constants

STOPWORDS = {
    "hospital",
    "medical",
    "center",
    "healthcare",
    "health",
    "services",
    "the",
    "and",
    "of",
}


class FacilityMatchResult(StrEnum):
    """Enumeration of possible facility matching outcomes, ordered by preference/confidence."""

    # Highest confidence - exact name match
    EXACT_MATCH = "Exact match"

    # High confidence - matched known abbreviation pattern
    ABBREVIATION_MATCH = "Abbreviation match"

    # Medium-high confidence - high similarity score
    FUZZY_MATCH = "Fuzzy match"

    # Medium confidence - single result from TF-IDF analysis
    SINGLE_MATCH = "Single match"

    # Lower confidence - multiple viable options, user needs to choose
    MULTIPLE_MATCHES = "Multiple matches"

    # Low confidence - too many results to be useful
    TOO_MANY_MATCHES = "Too many matches"

    # No confidence - no matching facilities found
    NO_MATCH = "No match"

    @property
    def preference_order(self) -> int:  # noqa: PLR0911
        """Returns the preference order for this match result (lower number = higher preference)."""
        match self:
            case FacilityMatchResult.EXACT_MATCH:
                return 1
            case FacilityMatchResult.ABBREVIATION_MATCH:
                return 2
            case FacilityMatchResult.FUZZY_MATCH:
                return 3
            case FacilityMatchResult.SINGLE_MATCH:
                return 4
            case FacilityMatchResult.MULTIPLE_MATCHES:
                return 5
            case FacilityMatchResult.TOO_MANY_MATCHES:
                return 6
            case FacilityMatchResult.NO_MATCH:
                return 7
            case _:
                raise ValueError(f"Invalid facility match result: {self}")


class FacilityMatch(BaseModel):
    """Represents the result of a facility matching operation."""

    result: FacilityMatchResult = Field(description="Result of the facility matching")
    matches: list[tuple[str, str, str]] = Field(
        description="List of all matched facilities"
    )

    def __str__(self):
        """Provides a human-readable string representation of the match result."""
        if self.result == FacilityMatchResult.NO_MATCH:
            return "No matches found."
        if self.result == FacilityMatchResult.TOO_MANY_MATCHES:
            return "Too many matches."
        if self.result == FacilityMatchResult.MULTIPLE_MATCHES:
            return "Multiple matches:\n" + "\n".join(
                [f"{name} ({c}, {s})" for name, c, s in self.matches]
            )
        return f"{self.result.value}: {self.matches[0][0]}, {self.matches[0][1]}, {self.matches[0][2]}"


def _check_exact_match(
    search_term_lower: str, city: str | None, state_abbrev: str | None
) -> FacilityMatch | None:
    """
    Checks for an exact match of the facility name, optionally filtered by city and state.

    Args:
        search_term_lower: The facility name to search for, in lowercase.
        city: Optional city to filter by.
        state_abbrev: Optional state abbreviation to filter by.

    Returns:
        A FacilityMatch object if an exact match is found, otherwise None.

    """
    for name, c, s in constants.FACILITY_LIST:
        if city and city.lower() != c.lower():
            continue
        if state_abbrev and state_abbrev.lower() != s.lower():
            continue
        if search_term_lower == name.lower():
            return FacilityMatch(
                result=FacilityMatchResult.EXACT_MATCH, matches=[(name, c, s)]
            )
    return None


def _check_abbreviation_match(  # noqa: C901
    search_term: str, city: str | None, state_abbrev: str | None
) -> FacilityMatch | None:
    """
    Checks for a match based on common abbreviation patterns of the facility name.

    Args:
        search_term: The abbreviation to search for.
        city: Optional city to filter by.
        state_abbrev: Optional state abbreviation to filter by.

    Returns:
        A FacilityMatch object if an abbreviation match is found, otherwise None.

    """
    if not (len(search_term) >= 2 and search_term.isalpha()):
        return None

    search_term_upper = search_term.upper()
    # skip_words definition moved here from original loop for minor optimization
    skip_words = {"of", "and", "the", "in", "at", "to", "for"}

    for name, c, s in constants.FACILITY_LIST:
        if city and city.lower() != c.lower():
            continue
        if state_abbrev and state_abbrev.lower() != s.lower():
            continue

        abbr1 = "".join(word[0].upper() for word in name.split() if word)
        if search_term_upper == abbr1:
            return FacilityMatch(
                result=FacilityMatchResult.ABBREVIATION_MATCH, matches=[(name, c, s)]
            )

        abbr2 = "".join(
            word[0].upper()
            for word in name.split()
            if word and word.lower() not in skip_words
        )
        if search_term_upper == abbr2:
            return FacilityMatch(
                result=FacilityMatchResult.ABBREVIATION_MATCH, matches=[(name, c, s)]
            )

        if "/" in name:
            parts = name.split("/")
            abbr3_simple = "".join(
                part.strip()[0].upper() for part in parts if part.strip()
            )
            if abbr3_simple and search_term_upper == abbr3_simple:
                return FacilityMatch(
                    result=FacilityMatchResult.ABBREVIATION_MATCH,
                    matches=[(name, c, s)],
                )

            abbr3_expanded = ""
            for part in parts:
                abbr3_expanded += "".join(
                    word[0].upper() for word in part.split() if word
                )
            if abbr3_expanded and search_term_upper == abbr3_expanded:
                return FacilityMatch(
                    result=FacilityMatchResult.ABBREVIATION_MATCH,
                    matches=[(name, c, s)],
                )
    return None


def _check_simple_fuzzy_match(
    search_term_lower: str, city: str | None, state_abbrev: str | None
) -> FacilityMatch | None:
    """
    Checks for a fuzzy match (high similarity) of the facility name.

    Args:
        search_term_lower: The facility name to search for, in lowercase.
        city: Optional city to filter by.
        state_abbrev: Optional state abbreviation to filter by.

    Returns:
        A FacilityMatch object if a fuzzy match is found (similarity >= 0.8), otherwise None.

    """
    for name, c, s in constants.FACILITY_LIST:
        if city and city.lower() != c.lower():
            continue
        if state_abbrev and state_abbrev.lower() != s.lower():
            continue

        similarity = SequenceMatcher(None, search_term_lower, name.lower()).ratio()
        if similarity >= 0.8:
            return FacilityMatch(
                result=FacilityMatchResult.FUZZY_MATCH, matches=[(name, c, s)]
            )
    return None


def _prepare_filtered_documents_and_entities(
    city: str | None, state_abbrev: str | None
) -> tuple[list[list[str]], list[tuple[str, str, str]], bool]:
    """
    Filters facilities by city/state and tokenizes their names for TF-IDF.

    Args:
        city: Optional city to filter by.
        state_abbrev: Optional state abbreviation to filter by.

    Returns:
        A tuple containing:
            - A list of tokenized facility names (documents).
            - A list of filtered facility entities (name, city, state).
            - A boolean indicating if any entities matched the city/state criteria.

    """
    documents = []
    filtered_entities = []
    has_matching_entities = False

    for name, c, s in constants.FACILITY_LIST:
        if city and city.lower() != c.lower():
            continue
        if state_abbrev and state_abbrev.lower() != s.lower():
            continue

        has_matching_entities = True
        filtered_entities.append((name, c, s))
        tokens = [
            word.lower()
            for word in re.findall(r"\b\w+\b", name.lower())
            if word.lower() not in STOPWORDS
        ]
        documents.append(tokens)
    return documents, filtered_entities, has_matching_entities


def _calculate_tfidf_based_matches(
    search_term: str,
    documents: list[list[str]],
    filtered_entities: list[tuple[str, str, str]],
) -> list[tuple[str, str, str, float]]:
    """
    Calculates TF-IDF scores and combines them with string similarity to find matches.

    Args:
        search_term: The original facility name search term.
        documents: A list of tokenized facility names.
        filtered_entities: A list of facility entities corresponding to the documents.

    Returns:
        A list of tuples, each containing (name, city, state, combined_score),
        for matches with a combined score >= 0.3.

    """

    term_df: defaultdict[str, int] = defaultdict(int)
    for doc in documents:
        for term in set(doc):
            term_df[term] += 1

    total_docs = max(1, len(documents))
    doc_vectors = []
    for doc in documents:
        term_freq = Counter(doc)
        doc_vector = {}
        for term, freq in term_freq.items():
            tf = freq / max(1, len(doc))
            idf = math.log((total_docs + 1) / (term_df.get(term, 0) + 1))
            doc_vector[term] = tf * idf
        doc_vectors.append(doc_vector)

    # Tokenize search_term for TF-IDF query vector
    query_tokens = [
        word.lower()
        for word in re.findall(
            r"\b\w+\b", search_term.lower()
        )  # Use search_term.lower() for consistency
        if word.lower() not in STOPWORDS
    ]
    if not query_tokens:
        return []

    query_tf = Counter(query_tokens)
    query_vector = {}
    for term, freq in query_tf.items():
        tf = freq / max(1, len(query_tokens))
        idf = math.log((total_docs + 1) / (term_df.get(term, 0) + 1))
        query_vector[term] = tf * idf

    scored_matches = []
    for i, doc_vector in enumerate(doc_vectors):
        dot_product = sum(
            query_vector.get(term, 0) * doc_vector.get(term, 0)
            for term in set(query_vector) | set(doc_vector)
        )
        query_magnitude = math.sqrt(
            max(0.0001, sum(val**2 for val in query_vector.values()))
        )
        doc_magnitude = math.sqrt(
            max(0.0001, sum(val**2 for val in doc_vector.values()))
        )

        similarity = 0.0
        if query_magnitude > 0 and doc_magnitude > 0:
            similarity = dot_product / (query_magnitude * doc_magnitude)

        name, fac_city, fac_state = filtered_entities[i]
        string_similarity = SequenceMatcher(None, search_term, name.lower()).ratio()
        combined_score = (0.7 * similarity) + (0.3 * string_similarity)

        if combined_score >= 0.3:
            scored_matches.append((name, fac_city, fac_state, combined_score * 100))

    return scored_matches


def _finalize_matches(
    scored_matches_with_score: list[tuple[str, str, str, float]],
) -> FacilityMatch:
    """
    Sorts TF-IDF based matches, removes duplicates, and determines the final match result.

    Args:
        scored_matches_with_score: A list of tuples (name, city, state, score).

    Returns:
        A FacilityMatch object representing the outcome (SINGLE_MATCH, MULTIPLE_MATCHES,
        TOO_MANY_MATCHES, or NO_MATCH).

    """
    sorted_matches_with_score = sorted(
        scored_matches_with_score,
        key=lambda item: item[3],
        reverse=True,
    )

    unique_matches_tuples = []
    seen_facilities = set()
    for name, c, s, _score in sorted_matches_with_score:
        facility_tuple = (name, c, s)
        if facility_tuple not in seen_facilities:
            seen_facilities.add(facility_tuple)
            unique_matches_tuples.append(facility_tuple)

    if len(unique_matches_tuples) > 5:
        return FacilityMatch(result=FacilityMatchResult.TOO_MANY_MATCHES, matches=[])
    if len(unique_matches_tuples) > 1:
        return FacilityMatch(
            result=FacilityMatchResult.MULTIPLE_MATCHES, matches=unique_matches_tuples
        )
    if len(unique_matches_tuples) == 1:
        return FacilityMatch(
            result=FacilityMatchResult.SINGLE_MATCH, matches=unique_matches_tuples
        )
    return FacilityMatch(result=FacilityMatchResult.NO_MATCH, matches=[])


def match_facility(
    facility_name: str, city: str | None = None, state_abbrev: str | None = None
) -> FacilityMatch:
    """
    Match a facility name against the list of known facilities.
    Uses TF-IDF for determining word importance and fuzzy matching.

    Args:
        facility_name: The name to search for
        city: Optional city to narrow down results
        state_abbrev: Optional two-letter state abbreviation to narrow down results

    Returns:
        A list of matching facilities, one per line, or a message if no matches

    """
    search_term = facility_name.strip()
    search_term_lower = search_term.lower()

    # If search_term is empty after strip, subsequent logic will correctly yield NO_MATCH
    # via empty query_tokens in _calculate_tfidf_based_matches.

    match_result = _check_exact_match(search_term_lower, city, state_abbrev)
    if match_result:
        return match_result

    match_result = _check_abbreviation_match(search_term, city, state_abbrev)
    if match_result:
        return match_result

    match_result = _check_simple_fuzzy_match(search_term_lower, city, state_abbrev)
    if match_result:
        return match_result

    # TF-IDF based matching
    documents, filtered_entities, has_matching_entities = (
        _prepare_filtered_documents_and_entities(city, state_abbrev)
    )

    if (not has_matching_entities) or (not documents):
        return FacilityMatch(result=FacilityMatchResult.NO_MATCH, matches=[])

    scored_matches = _calculate_tfidf_based_matches(
        search_term, documents, filtered_entities
    )

    return _finalize_matches(scored_matches)
