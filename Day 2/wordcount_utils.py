from pyspark.sql.functions import regexp_replace, trim, col, lower
def removePunctuation(column):
    """Removes punctuation, changes to lower case, and strips leading and trailing spaces."""
    return trim(lower(regexp_replace(column, "[^A-Za-z0-9 ]", ""))).alias("sentence")