from datetime import datetime
from repobird_leadgen.models import ContactInfo, RepoSummary

# Tests for ContactInfo


def test_contactinfo_instantiation_all_fields():
    """Test ContactInfo instantiation with all fields provided."""
    emails = ["test@example.com", "another@example.org"]
    twitter = "test_twitter"
    blog = "https://example.blog"
    contact = ContactInfo(emails=emails, twitter=twitter, blog=blog)
    assert contact.emails == emails
    assert contact.twitter == twitter
    assert contact.blog == blog


def test_contactinfo_instantiation_required_only():
    """Test ContactInfo instantiation with only default fields (empty list for emails)."""
    contact = ContactInfo()
    assert contact.emails == []
    assert contact.twitter is None
    assert contact.blog is None


def test_contactinfo_instantiation_some_optional():
    """Test ContactInfo instantiation with some optional fields."""
    emails = ["test@example.com"]
    twitter = "test_twitter"
    contact = ContactInfo(emails=emails, twitter=twitter)
    assert contact.emails == emails
    assert contact.twitter == twitter
    assert contact.blog is None


# Tests for RepoSummary


def test_reposummary_instantiation_required_only():
    """Test RepoSummary instantiation with only required fields."""
    now = datetime.now()
    summary = RepoSummary(
        full_name="octocat/Hello-World",
        description="My first repository on GitHub!",
        stars=100,
        language="Python",
        open_issues=10,
        good_first_issues=2,
        help_wanted_issues=3,
        last_push=now,
        contact=ContactInfo(),  # Provide a default ContactInfo
    )
    assert summary.full_name == "octocat/Hello-World"
    assert summary.description == "My first repository on GitHub!"
    assert summary.stars == 100
    assert summary.language == "Python"
    assert summary.open_issues == 10
    assert summary.good_first_issues == 2
    assert summary.help_wanted_issues == 3
    assert summary.last_push == now
    assert isinstance(summary.contact, ContactInfo)
    assert summary.contact.emails == []


def test_reposummary_instantiation_with_contact():
    """Test RepoSummary instantiation with a specific ContactInfo."""
    now = datetime.now()
    contact_details = ContactInfo(emails=["lead@example.com"], twitter="lead_handle")
    summary = RepoSummary(
        full_name="test/repo",
        description="A test repo",
        stars=5,
        language=None,
        open_issues=1,
        good_first_issues=0,
        help_wanted_issues=1,
        last_push=now,
        contact=contact_details,
    )
    assert summary.full_name == "test/repo"
    assert summary.description == "A test repo"
    assert summary.stars == 5
    assert summary.language is None
    assert summary.open_issues == 1
    assert summary.good_first_issues == 0
    assert summary.help_wanted_issues == 1
    assert summary.last_push == now
    assert summary.contact == contact_details
    assert summary.contact.emails == ["lead@example.com"]
    assert summary.contact.twitter == "lead_handle"
    assert summary.contact.blog is None
