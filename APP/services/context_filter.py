from difflib import SequenceMatcher


def remove_duplicate_chunks(chunks, similarity_threshold=0.92):
    """
    Remove highly similar retrieved chunks while
    preserving different information.
    """

    unique_chunks = []

    for chunk in chunks:

        duplicate = False

        for existing in unique_chunks:

            similarity = SequenceMatcher(
                None,
                chunk,
                existing
            ).ratio()

            if similarity >= similarity_threshold:
                duplicate = True
                break

        if not duplicate:
            unique_chunks.append(chunk)

    return unique_chunks