# Each rule is a tuple of:
# (suffix_to_strip, suffix_to_add_for_lemma, gender, number, case)
# All suffixes are Unicode Devanagari matras/characters

PARADIGM_TABLES = {

    # ----------------------------------------------------------------
    # MASCULINE CLASSES
    # ----------------------------------------------------------------

    'M1': [
        # लड़का class — masculine -ā final
        # Direct singular: लड़का (citation form, no change)
        ('ा', 'ा', 'M', 'SG', 'DIR'),
        # Direct plural: लड़के
        ('े', 'ा', 'M', 'PL', 'DIR'),
        # Oblique singular: लड़के (syncretism with dir pl)
        ('े', 'ा', 'M', 'SG', 'OBL'),
        # Oblique plural: लड़कों
        ('ों', 'ा', 'M', 'PL', 'OBL'),
    ],

    'M2': [
        # घर class — masculine consonant-final, largely invariant
        # Direct singular: घर (no suffix change)
        ('', '', 'M', 'SG', 'DIR'),
        # Direct plural: घर (same form — syncretism)
        ('', '', 'M', 'PL', 'DIR'),
        # Oblique singular: घर (same form — syncretism)
        ('', '', 'M', 'SG', 'OBL'),
        # Oblique plural: घरों
        ('ों', '', 'M', 'PL', 'OBL'),
    ],

    'M3': [
        # आदमी class — masculine -ī final, invariant except obl pl
        # Direct singular: आदमी
        ('ी', 'ी', 'M', 'SG', 'DIR'),
        # Direct plural: आदमी (same form)
        ('ी', 'ी', 'M', 'PL', 'DIR'),
        # Oblique singular: आदमी (same form)
        ('ी', 'ी', 'M', 'SG', 'OBL'),
        # Oblique plural: आदमियों
        ('ियों', 'ी', 'M', 'PL', 'OBL'),
    ],

    'M4': [
        # आलू class — masculine -ū/-u final
        # Direct singular: आलू
        ('ू', 'ू', 'M', 'SG', 'DIR'),
        # Direct plural: आलू (same form)
        ('ू', 'ू', 'M', 'PL', 'DIR'),
        # Oblique singular: आलू (same form)
        ('ू', 'ू', 'M', 'SG', 'OBL'),
        # Oblique plural: आलुओं
        ('ुओं', 'ू', 'M', 'PL', 'OBL'),
    ],

    # ----------------------------------------------------------------
    # FEMININE CLASSES
    # ----------------------------------------------------------------

    'F1': [
        # लड़की class — feminine -ī final
        # Direct singular: लड़की
        ('ी', 'ी', 'F', 'SG', 'DIR'),
        # Direct plural: लड़कियाँ
        ('ियाँ', 'ी', 'F', 'PL', 'DIR'),
        # Oblique singular: लड़की (same as dir sg)
        ('ी', 'ी', 'F', 'SG', 'OBL'),
        # Oblique plural: लड़कियों
        ('ियों', 'ी', 'F', 'PL', 'OBL'),
    ],

    'F2': [
        # आशा class — feminine -ā final
        # Direct singular: आशा
        ('ा', 'ा', 'F', 'SG', 'DIR'),
        # Direct plural: आशाएँ
        ('ाएँ', 'ा', 'F', 'PL', 'DIR'),
        # Oblique singular: आशा (same as dir sg)
        ('ा', 'ा', 'F', 'SG', 'OBL'),
        # Oblique plural: आशाओं
        ('ाओं', 'ा', 'F', 'PL', 'OBL'),
    ],

    'F3': [
        # रात class — feminine consonant-final
        # Direct singular: रात
        ('', '', 'F', 'SG', 'DIR'),
        # Direct plural: रातें
        ('ें', '', 'F', 'PL', 'DIR'),
        # Oblique singular: रात (same as dir sg)
        ('', '', 'F', 'SG', 'OBL'),
        # Oblique plural: रातों
        ('ों', '', 'F', 'PL', 'OBL'),
    ],

    'F4': [
        # शान्ति class — feminine -i final
        # Direct singular: शान्ति
        ('ि', 'ि', 'F', 'SG', 'DIR'),
        # Direct plural: शान्तियाँ
        ('ियाँ', 'ि', 'F', 'PL', 'DIR'),
        # Oblique singular: शान्ति (same as dir sg)
        ('ि', 'ि', 'F', 'SG', 'OBL'),
        # Oblique plural: शान्तियों
        ('ियों', 'ि', 'F', 'PL', 'OBL'),
    ],

    'F5': [
        # वायु class — feminine -u/-ū final
        # Direct singular: वायु
        ('ु', 'ु', 'F', 'SG', 'DIR'),
        # Direct plural: वायुएँ
        ('ुएँ', 'ु', 'F', 'PL', 'DIR'),
        # Oblique singular: वायु (same as dir sg)
        ('ु', 'ु', 'F', 'SG', 'OBL'),
        # Oblique plural: वायुओं
        ('ुओं', 'ु', 'F', 'PL', 'OBL'),
    ],
}