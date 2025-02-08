# occam_vb/core.py

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Set, Union
from dataclasses import dataclass, field
import scipy.stats as stats
from collections import defaultdict
import yaml
from pathlib import Path


@dataclass
class VariableDefinition:
    """Represents a single variable definition from YAML"""
    name: str            # Full name (e.g., "APOE") - only for display
    abbrev: str         # Abbreviation (e.g., "Ap") - used internally
    cardinality: int    # Number of possible values
    type: int           # 0=ignore, 1=IV, 2=DV
    rebin: Optional[str] = None  # Optional rebinning specification

    def __post_init__(self):
        """Validate variable definition"""
        # Validate type
        if self.type not in [0, 1, 2]:
            raise ValueError(f"Invalid variable type {self.type}")
            
        # Validate cardinality
        if self.cardinality < 1:
            raise ValueError(f"Invalid cardinality {self.cardinality}")
            
        # Validate abbreviation format
        if not self.abbrev.isalpha():
            raise ValueError("Abbreviation must contain only letters")

@dataclass
class VariableList:
    """Handles variable definitions and mappings"""
    variables: Dict[str, VariableDefinition] = field(default_factory=dict)
    _dv: Optional[str] = field(init=False, default=None)
    _ivs: List[str] = field(init=False, default_factory=list)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'VariableList':
        """Create VariableList from YAML variable definitions file"""
        with open(yaml_path) as f:
            config = yaml.safe_load(f)
            
        varlist = cls()
        for name, info in config['variables'].items():
            vardef = VariableDefinition(
                name=name,
                abbrev=info['abbrev'],
                cardinality=info['cardinality'],
                type=info['type'],
                rebin=info.get('rebin')
            )
            varlist.variables[vardef.abbrev] = vardef
            
            if vardef.type == 2:  # DV
                if varlist._dv is not None:
                    raise ValueError("Multiple DVs not supported")
                varlist._dv = vardef.abbrev
            elif vardef.type == 1:  # IV
                varlist._ivs.append(vardef.abbrev)
                
        return varlist

    def get_active_abbrevs(self) -> List[str]:
        """Get abbreviations of all active (non-ignored) variables"""
        return [abbrev for abbrev, var in self.variables.items() 
                if var.type != 0]  # type 0 means ignored
                
    def get_dv_abbrev(self) -> str:
        """Get abbreviation of DV"""
        if self._dv is None:
            raise ValueError("No DV found")
        return self._dv
        
    def get_iv_abbrevs(self) -> List[str]:
        """Get abbreviations of all IVs"""
        return self._ivs.copy()
        
    def get_full_name(self, abbrev: str) -> str:
        """Get full name for display purposes"""
        return self.variables[abbrev].name
    def verify_model_spec(self, model_spec: str):
        """Verify that a model specification is valid"""
        if not model_spec:
            raise ValueError("Empty model specification")
            
        # Split into components
        components = model_spec.split(':')
        
        # Track variables used
        used_vars = set()
        
        for comp in components:
            if comp == 'IV':
                # IV component should contain all IVs
                used_vars.update(self.get_iv_abbrevs())
            else:
                # Parse component into variables
                vars = self._parse_component(comp)
                used_vars.update(vars)
                
                # Verify DV is present
                if self.get_dv_abbrev() not in vars:
                    raise ValueError(f"Component {comp} missing DV")
                    
        # Verify all active variables are used
        all_vars = set(self.get_active_abbrevs())
        if not used_vars == all_vars:
            missing = all_vars - used_vars
            raise ValueError(f"Model missing variables: {missing}")

    def _parse_component(self, comp: str) -> List[str]:
        """Parse component string into list of variable abbreviations"""
        vars = []
        current = []
        
        for c in comp:
            if c == 'Z':  # DV
                if current:
                    vars.append(''.join(current))
                    current = []
                vars.append(self.get_dv_abbrev())
            else:
                current.append(c)
                
        if current:
            vars.append(''.join(current))
            
        return vars        

    def add_variable(self, vardef: VariableDefinition):
        """Add a variable definition"""
        # Check for duplicate abbreviations
        if vardef.abbrev in self.variables:
            raise ValueError(f"Duplicate abbreviation {vardef.abbrev}")
            
        # Add to variables dict
        self.variables[vardef.abbrev] = vardef
        
        # Update DV/IV tracking 
        if vardef.type == 2:  # DV
            if self._dv is not None:
                raise ValueError("Multiple DVs not supported")
            self._dv = vardef.abbrev
        elif vardef.type == 1:  # IV
            self._ivs.append(vardef.abbrev)      

            
@dataclass(frozen=True)
class State:
    """Represents an immutable state tuple with variables""" 
    values: Tuple[Union[int, str], ...]  # Allow both ints and strings (for missing '.')
    varlist: VariableList
    subset_vars: Optional[List[str]] = None  # Add this field for projected states
    _active_vars: List[str] = field(init=False, default_factory=list)
    
    def __post_init__(self):
        """Validate state against variable list and set active vars"""
        if not isinstance(self.varlist, VariableList):
            raise TypeError(f"varlist must be VariableList instance, got {type(self.varlist)}")
            
        # Initialize _active_vars based on subset_vars or all active vars
        active_vars = []
        if self.subset_vars:
            # Verify all subset vars are valid
            for var in self.subset_vars:
                if var not in self.varlist.variables:
                    raise ValueError(f"Invalid variable {var} in subset_vars")
            active_vars = self.subset_vars
        else:
            active_vars = self.varlist.get_active_abbrevs()
            
        object.__setattr__(self, '_active_vars', active_vars)
                          
        if len(self.values) != len(self._active_vars):
            raise ValueError(f"Values count ({len(self.values)}) must match variables ({len(self._active_vars)})")
            
    def __hash__(self):
        """Hash based on values and variable names"""
        return hash((self.values, tuple(self._active_vars)))
        
    def __eq__(self, other):
        """Equality check for dictionary keys"""
        if not isinstance(other, State):
            return False
        return (self.values == other.values and 
                self._active_vars == other._active_vars)
    def __lt__(self, other: 'State') -> bool:
        """Less than comparison for sorting"""
        if not isinstance(other, State):
            return NotImplemented
            
        # Compare values first
        for v1, v2 in zip(self.values, other.values):
            # Handle missing values ('.')
            if v1 == '.' and v2 != '.':
                return True
            if v2 == '.' and v1 != '.':
                return False
            if v1 != v2:
                # Convert to strings for comparison if needed
                s1 = str(v1)
                s2 = str(v2)
                return s1 < s2
                
        # If values are equal, compare by variables
        for v1, v2 in zip(self._active_vars, other._active_vars):
            if v1 != v2:
                return v1 < v2
                
        # If everything is equal, default to False
        return False

    def __gt__(self, other: 'State') -> bool:
        """Greater than comparison"""
        if not isinstance(other, State):
            return NotImplemented
        return other < self

    def __le__(self, other: 'State') -> bool:
        """Less than or equal comparison"""
        if not isinstance(other, State):
            return NotImplemented
        return self < other or self == other

    def __ge__(self, other: 'State') -> bool:
        """Greater than or equal comparison"""
        if not isinstance(other, State):
            return NotImplemented
        return other <= self
    @property 
    def variables(self) -> List[str]:
        """Get variable abbreviations in order"""
        return self._active_vars.copy()

    def get_variable_values(self, vars: List[str]) -> List[Union[int, str]]:
        """Get values for specified variables"""
        return [self.values[self._active_vars.index(v)] for v in vars 
                if v in self._active_vars]

    def project(self, vars: List[str]) -> 'State':
        """Project state onto subset of variables"""
        # Verify all projection variables are valid
        proj_vars = [v for v in vars if v in self._active_vars]
        if not proj_vars:
            raise ValueError(f"No valid variables to project onto: {vars}")
            
        # Get values for projection variables
        proj_values = []
        for var in proj_vars:
            if var in self._active_vars:
                idx = self._active_vars.index(var)
                proj_values.append(self.values[idx])
                
        # Create new state with projected variables
        return State(tuple(proj_values), self.varlist, subset_vars=proj_vars)
        
@dataclass
        
@dataclass
class StateSpace:
    """Manages state frequencies and calculations"""
    data: pd.DataFrame
    varlist: VariableList
    dv_col: str
    iv_cols: List[str] = field(init=False)
    n: int = field(init=False)
    keysize: int = field(init=False)
    state_frequencies: Dict[State, float] = field(init=False)
    cache_projections: Dict[Tuple[str, ...], Dict[State, float]] = field(default_factory=dict)
    varcardinalities: Dict[str, int] = field(init=False)
    
    def __post_init__(self):
        """Initialize derived attributes"""
        # Get all active variables using abbreviations
        self.variables = self.varlist.get_active_abbrevs()
        self.iv_cols = self.varlist.get_iv_abbrevs()
        self.keysize = len(self.variables)
        self.n = len(self.data)
        
        # Calculate variable cardinalities
        self.varcardinalities = {
            var: self.varlist.variables[var].cardinality
            for var in self.variables
        }
        
        # Initialize frequency tables
        self.state_frequencies = self._compute_frequencies()

    def _compute_frequencies(self) -> Dict[State, float]:
        """Compute frequencies exactly as C++ does"""
        freqs = defaultdict(float)
        
        # Match C++ handling of missing values
        for _, row in self.data.iterrows():
            values = []
            for var in self.variables:
                val = row[var]
                # Match C++ missing value handling
                values.append(int(val) if pd.notna(val) else '.')
                
            state = State(
                values=tuple(values),
                varlist=self.varlist 
            )            
            freqs[state] += 1.0  # Always increment by 1.0 as C++ does
            
        return dict(freqs)   

    def project_frequencies(self, states: Dict[State, float], 
                          variables: List[str]) -> Dict[State, float]:
        """
        Project frequencies onto a subset of variables.
        Caches results for efficiency.
        """
        # Check cache first
        cache_key = tuple(sorted(variables))
        if cache_key in self.cache_projections:
            return self.cache_projections[cache_key]
            
        # Compute projection
        projected = defaultdict(float)
        for state, freq in states.items():
            proj_state = state.project(variables)
            projected[proj_state] += freq
            
        # Cache result
        result = dict(projected)
        self.cache_projections[cache_key] = result
        return result
        
    def get_variable_cardinality(self, var: str) -> int:
        """Get cardinality (number of possible values) for a variable"""
        return self.varcardinalities[var]
        
    def get_state_space_size(self) -> int:
        """
        Calculate total size of state space.
        This matches the C++ implementation's calculation.
        """
        size = 1
        for card in self.varcardinalities.values():
            size *= card
        return size
        
    def compute_df(self, relation_vars: List[str]) -> int:
        """
        Compute degrees of freedom for a relation.
        This matches the C++ implementation's calculation.
        
        Args:
            relation_vars: List of variable abbreviations in the relation
            
        Returns:
            Degrees of freedom for the relation
        """
        df = 1
        for var in relation_vars:
            df *= self.varcardinalities[var]
        return df - 1
        
    def compute_entropy(self, freqs: Dict[State, float], base: float = 2.0) -> float:
        """
        Compute entropy of a frequency distribution.
        
        Args:
            freqs: Dictionary mapping states to frequencies
            base: Logarithm base (default 2 for bits)
            
        Returns:
            Entropy value
        """
        total = sum(freqs.values())
        if total <= 0:
            return 0.0
            
        entropy = 0.0
        for freq in freqs.values():
            if freq > 0:
                prob = freq / total
                entropy -= prob * np.log(prob) / np.log(base)
                
        return entropy

    def compute_conditional_entropy(self, joint_freqs: Dict[State, float], 
                                  conditioning_vars: List[str]) -> float:
        """
        Compute conditional entropy H(Y|X).
        
        Args:
            joint_freqs: Joint frequency distribution
            conditioning_vars: Variables to condition on
            
        Returns:
            Conditional entropy value
        """
        # Project to get marginal distribution
        marg_freqs = self.project_frequencies(joint_freqs, conditioning_vars)
        
        h_cond = 0.0
        total = sum(joint_freqs.values())
        
        # For each state of conditioning variables
        for marg_state, marg_freq in marg_freqs.items():
            # Skip if marginal probability is 0
            if marg_freq <= 0:
                continue
                
            # Get conditional distribution
            cond_dist = {}
            marg_prob = marg_freq / total
            
            for joint_state, freq in joint_freqs.items():
                if joint_state.project(conditioning_vars) == marg_state:
                    cond_dist[joint_state] = freq / marg_freq
                    
            # Add weighted conditional entropy
            if cond_dist:
                h_cond += marg_prob * self.compute_entropy(cond_dist)
                
        return h_cond

    def is_missing_value(self, value) -> bool:
        """Check if a value represents missing data"""
        return value == '.' or pd.isna(value)
        
    def clear_cache(self):
        """Clear projection cache"""
        self.cache_projections.clear()
        
    def compute_reference_entropies(self) -> Tuple[float, float, float]:
        """
        Compute reference entropies (top, bottom, independent).
        Match C++ VBMManager.cpp computeH() method.
        """
        # Get top model entropy (data)
        all_vars = sorted(self.variables)  # Sort to match C++ order
        top_freqs = self.state_frequencies
        top_h = self.compute_entropy(top_freqs)
        
        # Get independent relation entropy
        iv_freqs = self.project_frequencies(self.state_frequencies, self.iv_cols)
        iv_h = self.compute_entropy(iv_freqs)
        
        # Get bottom model entropy
        # Project to get single variable distributions
        bottom_h = 0.0
        for var in self.variables:
            var_freqs = self.project_frequencies(self.state_frequencies, [var])
            bottom_h += self.compute_entropy(var_freqs)
        
        return top_h, bottom_h, iv_h

    def make_relation(self, variables: List[str]) -> Relation:
        """Create relation from list of variables, matching C++ VBMManager"""
        # Verify variables are valid
        for var in variables:
            if var not in self.variables:
                raise ValueError(f"Invalid variable {var}")
                
        # Create relation with sorted variables
        relation = Relation(
            variables=sorted(variables),  # Sort to match C++ ordering
            varlist=self.varlist
        )
        
        # Project frequencies onto relation variables 
        relation.table = self.project_frequencies(
            self.state_frequencies, 
            relation.variables
        )
        
        return relation

    def compute_relation_statistics(self, relation: Relation):
        """Compute entropy and df for a relation"""
        # Compute degrees of freedom if not cached
        if 'df' not in relation.attributes:
            df = 1  # Start with 1 as in C++
            for var in relation.variables:
                df *= self.varcardinalities[var]
            df = df - 1  # Subtract 1 for constraint
            relation.attributes['df'] = df
            
        # Compute entropy if not cached
        if 'h' not in relation.attributes:
            total = sum(relation.table.values())
            if total <= 0:
                relation.attributes['h'] = 0.0
            else:
                h = 0.0
                for freq in relation.table.values():
                    if freq > 0:
                        prob = freq / total
                        h -= prob * np.log2(prob)
                relation.attributes['h'] = h

    def get_relation_projection(self, relation: Relation) -> Dict[State, float]:
        """Get frequency projection for a relation, using cache"""
        # Use sorted variable tuple as cache key 
        cache_key = tuple(sorted(relation.variables))
        
        if cache_key not in self.cache_projections:
            # Project frequencies
            freqs = self.project_frequencies(
                self.state_frequencies,
                relation.variables
            )
            self.cache_projections[cache_key] = freqs
            
        return self.cache_projections[cache_key]        

@dataclass
class Model:
    """
    Represents a reconstructability model built from relations.
    Mirrors C++ implementation exactly.
    """
    components: List[List[str]] = field(default_factory=list)  # Lists of variable abbreviations
    variables: List[str]         # All variable abbreviations
    dv_col: str                 # DV abbreviation
    state_space: StateSpace     # Reference to state space object
    statistics: Dict[str, float] = field(default_factory=dict)
    id: Optional[int] = None
    level: Optional[int] = None
    progenitor: Optional['Model'] = None
    incremental_alpha: float = 0.0
    progenitor_id: Optional[int] = None
    
    def __post_init__(self):
        """Initialize model and validate components"""
        if not self.components:
            raise ValueError("Model must have at least one component")
                
        # Verify all variables used are valid
        all_vars = set(self.variables)
        for comp in self.components:
            if not set(comp).issubset(all_vars):
                raise ValueError(f"Invalid variables in component {comp}")
            
        # For directed systems, verify DV in predicting components
        if self.dv_col:
            for comp in self.components[1:]:  # Skip IV component
                if self.dv_col not in comp:
                    raise ValueError(f"Component {comp} missing DV")
                    
        # Cache computed values
        self._cache = {}
        
    def compute_component_df(self, comp: List[str]) -> int:
        """Compute degrees of freedom for a component"""
        df = 1  # Start with 1 as in C++
        for var in comp:
            df *= self.state_space.get_variable_cardinality(var)
        return df - 1  # Subtract 1 for constraint

    def compute_component_entropy(self, comp: List[str], freqs: Dict[State, float]) -> float:
        """Compute entropy for a component"""
        # Project frequencies to component variables
        proj_freqs = self.state_space.project_frequencies(freqs, comp)
        
        total = sum(proj_freqs.values())
        if total <= 0:
            return 0.0
            
        h = 0.0
        for freq in proj_freqs.values():
            if freq > 0:
                prob = freq / total 
                h -= prob * np.log2(prob)
        return h

    def add_relation(self, relation: Relation):
        """Add relation to model"""
        self.relations.append(relation)
        # Clear caches since model changed
        self._cache.clear()
        self.statistics.clear()
        
    def get_relation(self, index: int) -> Relation:
        """Get relation by index"""
        return self.relations[index]
        
    def get_relation_count(self) -> int:
        """Get number of relations"""
        return len(self.relations)
        
    def get_name(self) -> str:
        """Get model name in OCCAM format"""
        parts = []
        iv_vars = set(v for v in self.variables if v != self.dv_col)
        
        for rel in self.relations:
            if set(rel.variables) == iv_vars:
                parts.append("IV")
            else:
                parts.append(rel.get_print_name())
        return ":".join(parts)

    def get_predicting_components(self) -> List[Relation]:
        """Get relations containing DV"""
        return [rel for rel in self.relations 
                if rel.has_variable(self.dv_col)]
                
    def get_iv_component(self) -> Optional[Relation]:
        """Get IV relation if present"""
        iv_vars = set(v for v in self.variables if v != self.dv_col)
        for rel in self.relations:
            if set(rel.variables) == iv_vars:
                return rel
        return None

    def is_loopless(self) -> bool:
        """Check if model is loopless"""
        # For directed systems, loopless means single predicting component
        if self.dv_col:
            pred_comps = self.get_predicting_components()
            return len(pred_comps) == 1
            
        # For neutral systems, check for variable overlap
        for i, rel1 in enumerate(self.relations):
            for rel2 in self.relations[i+1:]:
                if len(set(rel1.variables) & set(rel2.variables)) > 1:
                    return False
        return True

    def compute_df(self) -> int:
        """Compute degrees of freedom"""
        if 'df' in self._cache:
            return self._cache['df']
            
        # Start with -1 to account for normalization constraint
        df = -1
        for rel in self.relations:
            # Add df for each relation
            df += rel.compute_df(self.state_space)
            
        self._cache['df'] = df
        return df

    def compute_entropy(self) -> float:
        """
        Compute entropy using relations.
        Matches C++ VBMManager implementation exactly.
        """
        if 'h' in self._cache:
            return self._cache['h']
            
        # Compute entropy for each relation
        h = 0.0
        for rel in self.relations:
            h += rel.compute_entropy(self.state_space)
            
        self._cache['h'] = h
        return h

    def get_fitted_probs(self) -> Dict[State, float]:
        """Get fitted probabilities using IPF"""
        # Get frequencies projected onto each component 
        comp_tables = []
        for comp in self.components:
            comp_tables.append(self.state_space.project_frequencies(
                self.state_space.state_frequencies,
                comp
            ))
                
        # Initialize with all states from frequency data
        states = sorted(list(self.state_space.state_frequencies.keys()))
        
        # Initialize uniform distribution    
        q = {state: 1.0/len(states) for state in states}
         
        # IPF main loop - Modified to match C++ exactly
        MAX_ITERATIONS = 266  # From C++ defaults
        EPSILON = 0.25  # From C++ ipf-maxdev
        iteration = 0
        
        while iteration < MAX_ITERATIONS:
            max_dev = 0.0
            
            # Iterate through components
            for i, comp in enumerate(self.components):
                # Project current q onto component variables
                q_proj = defaultdict(float)
                for state, prob in q.items():
                    proj_state = state.project(comp)
                    q_proj[proj_state] += prob
                        
                # Update q based on ratio - Match C++ calculation
                for state, prob in q.items():
                    proj_state = state.project(comp)
                    if prob > 0 and q_proj[proj_state] > 0 and proj_state in comp_tables[i]:
                        mult = comp_tables[i][proj_state] / q_proj[proj_state]
                        diff = abs((mult - 1.0) * prob)  # Match C++ calculation
                        q[state] *= mult
                        max_dev = max(max_dev, diff)
                
                # Normalize after each component (match C++)
                total = sum(q.values())
                if total > 0:
                    q = {state: prob/total for state, prob in q.items()}
                    
            iteration += 1
            if max_dev <= EPSILON:
                break
                
        return q

    def set_progenitor(self, model: 'Model'):
        """Set progenitor and compute incremental alpha"""
        self.progenitor = model
        self.progenitor_id = model.id
        
        # Compute incremental alpha between this model and progenitor
        self.incremental_alpha = self._compute_incremental_alpha(model)
        
    def _compute_incremental_alpha(self, progenitor: 'Model') -> float:
        """Compute incremental alpha between this model and progenitor"""
        # Get change in df and likelihood ratio
        ddf = abs(self.compute_df() - progenitor.compute_df())
        dlr = abs(self.statistics['dlr'] - progenitor.statistics['dlr'])
        
        # Compute p-value
        if ddf > 0:
            return 1 - stats.chi2.cdf(dlr, ddf)
        return 1.0

    def create_relations(self) -> List[Relation]:
        """Create relations from components, matching C++ VBMManager"""
        relations = []
        
        # First convert each component to a relation
        for comp in self.components:
            relation = self.state_space.make_relation(comp)
            relations.append(relation)
                
        return relations

    def compute_model_statistics(self, ref_model: Optional['Model'] = None):
        """
        Compute all model statistics, matching C++ VBMManager.
        """
        # Get relations if we haven't already
        if not hasattr(self, '_relations'):
            self._relations = self.create_relations()
                
        # Get reference model if needed
        if ref_model is None:
            # Create independence model as reference
            ref_components = [sorted(self.state_space.iv_cols), [self.dv_col]]
            ref_model = Model(
                components=ref_components,
                variables=self.variables,
                dv_col=self.dv_col,
                state_space=self.state_space
            )
        
        # Get reference relations
        ref_relations = ref_model.create_relations()
        
        # Compute statistics for each relation
        for rel in self._relations:
            self.state_space.compute_relation_statistics(rel)
            
        # Get fitted probabilities
        q = self.get_fitted_probs()
        
        # Calculate model entropy and df
        h = self.compute_entropy(q)
        df = sum(rel.attributes['df'] for rel in self._relations)
        
        # Get reference values 
        ref_q = ref_model.get_fitted_probs()
        ref_h = ref_model.compute_entropy(ref_q)
        ref_df = sum(rel.attributes['df'] for rel in ref_relations)
        
        # Compute derived statistics exactly as C++ does
        ddf = abs(df - ref_df)
        dlr = 2.0 * self.state_space.n * (ref_h - h) * np.log(2)
        
        # Alpha
        alpha = 1.0
        if ddf > 0:
            alpha = 1 - stats.chi2.cdf(dlr, ddf)
            
        # Information
        top_h, bottom_h, iv_h = self.state_space.compute_reference_entropies()
        if abs(bottom_h - top_h) > 1e-10:
            information = (bottom_h - h) / (bottom_h - top_h)
            information = max(0, min(1, information))
        else:
            information = 0.0
            
        # DV uncertainty reduction
        dh_dv = 0.0
        if self.dv_col:
            dv_obs = self.state_space.project_frequencies(
                self.state_space.state_frequencies, [self.dv_col])
            dv_bottom = self.state_space.project_frequencies(
                ref_q, [self.dv_col])
                
            h_dv = self.state_space.compute_entropy(dv_obs)
            h_dv_bottom = self.state_space.compute_entropy(dv_bottom)
            
            if h_dv_bottom > 0:
                dh_dv = 100.0 * (h_dv_bottom - h_dv) / h_dv_bottom
                
        # AIC and BIC
        aic = dlr - 2.0*ddf 
        bic = dlr - np.log(self.state_space.n)*ddf
        
        # Update model statistics
        self.statistics.update({
            'h': h,
            'df': df,
            'ddf': ddf,
            'dlr': dlr,
            'alpha': alpha,
            'information': information,
            'dh_dv': dh_dv,
            'aic': aic,
            'bic': bic
        })


class VariableNameMapper:
    """Maps between full variable names and their abbreviations"""
    
    def __init__(self, full_names: List[str], abbrevs: Optional[Dict[str, str]] = None):
        """
        Initialize mapper with full variable names and optional abbreviation mappings
        
        Args:
            full_names: List of full variable names
            abbrevs: Optional dict mapping full names to abbreviations
        """
        self.full_to_abbrev = {}
        self.abbrev_to_full = {}
        
        if abbrevs:
            # Use provided abbreviations
            for full_name, abbrev in abbrevs.items():
                if full_name not in full_names:
                    raise ValueError(f"Variable {full_name} not in data")
                self.full_to_abbrev[full_name] = abbrev
                self.abbrev_to_full[abbrev] = full_name
        else:
            # Auto-generate abbreviations
            for name in full_names:
                # Default to first character as abbreviation
                abbrev = name[0].upper()
                if abbrev in self.abbrev_to_full:
                    # If conflict, use first two characters
                    abbrev = name[:2].upper()
                self.full_to_abbrev[name] = abbrev
                self.abbrev_to_full[abbrev] = name
                
    def get_abbrev(self, full_name: str) -> str:
        """Get abbreviation for a full variable name"""
        return self.full_to_abbrev.get(full_name, full_name)
        
    def get_full_name(self, abbrev: str) -> str:
        """Get full name for an abbreviation"""
        return self.abbrev_to_full.get(abbrev, abbrev)
        
    def parse_component(self, comp_str: str, dv_col: str) -> List[str]:
        """Parse component string into list of full variable names"""
        vars = []
        current = []
        
        for c in comp_str:
            if c == 'Z':
                # Handle DV
                if current:
                    # Add accumulated variable if any
                    abbrev = ''.join(current)
                    vars.append(self.get_full_name(abbrev))
                    current = []
                vars.append(dv_col)
            else:
                current.append(c)
                
        if current:
            # Add any remaining variable
            abbrev = ''.join(current)
            vars.append(self.get_full_name(abbrev))
            
        return vars

@dataclass
class Occam:
    """Main OCCAM implementation integrating all components"""
    data: pd.DataFrame
    varlist: VariableList
    state_space: StateSpace = field(init=False)
    
    def __post_init__(self):
        """Initialize state space and verify data"""
        # Convert columns to use abbreviations  
        name_to_abbrev = {var.name: var.abbrev 
                         for var in self.varlist.variables.values()}
        self.data = self.data.rename(columns=name_to_abbrev)
        
        # Get all active variables using abbreviations
        self.all_vars = self.varlist.get_active_abbrevs()
        self.dv = self.varlist.get_dv_abbrev()
        self.ivs = self.varlist.get_iv_abbrevs()
        
        # Filter to active columns
        self.data = self.data[self.all_vars]
        
        # Initialize state space
        self.state_space = StateSpace(
            self.data,
            self.varlist,
            self.dv
        )
        
    def make_model(self, model_name: str) -> Model:
        """Create model from string specification matching C++ exactly"""
        # Validate model specification
        self.varlist.verify_model_spec(model_name)
        
        # Parse components
        components = []
        model_parts = model_name.split(':')
        
        # Always put IV component first for directed systems
        if self.dv:  # If we have a DV it's directed
            iv_comp = sorted(self.ivs)  # Sort IV names alphabetically
            components.append(iv_comp)
            
            # Handle remaining components
            for part in model_parts:
                if part != 'IV':  # Skip IV part since we added it
                    comp = self.varlist._parse_component(part)
                    if set(comp) != set(self.ivs):  # Skip if this is just IVs
                        components.append(sorted(comp))  # Sort variables in component
        else:
            # For neutral systems, just parse all components
            for part in model_parts:
                comp = self.varlist._parse_component(part)
                components.append(sorted(comp))
                
        # Sort remaining components in same order as C++
        if len(components) > 1:
            components[1:] = sorted(components[1:], 
                                  key=lambda x: (''.join(x), len(x)))
            
        # Create model
        return Model(
            components=components,
            variables=self.all_vars,
            dv_col=self.dv,
            state_space=self.state_space
        )

    def search(self, 
              width: int = 3, 
              levels: int = 7,
              criterion: str = 'h',  # Changed default from 'bic' to 'h'
              start_model: str = None,
              ref_model: str = None) -> List[Model]:
        """
        Perform model search.
        
        Args:
            width: Search width (models to keep per level)
            levels: Number of levels to search
            criterion: Sort criterion ('bic', 'aic', or 'information') 
            start_model: Starting model (default: independence)
            ref_model: Reference model (default: independence)
            
        Returns:
            List of models found during search
        """
        print("Option settings:")
        print(f"Search width: {width}")
        print(f"Search levels: {levels}")
        print(f"Search direction: up")
        print(f"Sort by: {criterion}")
        print(f"\nState Space Size: {self.state_space.get_state_space_size()}")
        print(f"Sample Size: {self.state_space.n}")

        # Set up initial models
        if not start_model:
            start_model = self._make_independence_model()
        else:
            start_model = self.make_model(start_model)
            
        if not ref_model:
            ref_model = self._make_independence_model()
        else:
            ref_model = self.make_model(ref_model)
            
        # Initialize start model
        start_model.level = 0
        start_model.id = 1
        
        # Initialize model lists
        models = [start_model]  # Current level models
        all_models = [start_model]  # All models found
        
        # Compute statistics for start model
        self._compute_model_statistics(start_model, ref_model)
        
        # Search through levels
        for level in range(levels):
            print(f"\nLevel {level + 1}:", end=' ')
            
            # Generate and evaluate candidates
            candidates = []
            for model in models:
                parents = self._generate_parents(model)
                candidates.extend(parents)
                
            if not candidates:
                print("No more candidates found")
                break
                
            # Remove duplicates
            candidates = self._remove_duplicate_models(candidates)
            total_models = len(all_models) + len(candidates)
            kept_models = len(all_models) + min(width, len(candidates))
            mem_usage = "0 kb"  # We could implement actual memory tracking
            print(f"{len(candidates)} models generated, {min(width, len(candidates))} kept; "
                  f"{total_models} total models, {kept_models} total kept; "
                  f"{mem_usage} memory used", end=' ')
            
            # Compute statistics and sort
            for model in candidates:
                model.level = level + 1
                self._compute_model_statistics(model, ref_model)
                
            # Sort by criterion
            candidates = self._sort_models(candidates, criterion)
            
            # Keep best models
            models = candidates[:width]
            
            # Assign IDs and track ancestry
            for i, model in enumerate(models):
                model.id = len(all_models) + i + 1
                
            all_models.extend(models)
            
            # Print best model at this level
            if models:
                best = models[0]
                print(f"\nBest: {best.get_name()} ({criterion}={best.statistics[criterion]:.4f})")

        # Sort by h and print final model report 
        models_by_h = sorted(all_models, key=lambda m: m.statistics['h'])
        print("\nID MODEL Level H dDF dLR Alpha Inf %dH(DV) dAIC dBIC Inc.Alpha Prog. %C(Data) %cover")
        for i, model in enumerate(models_by_h, 1):
            stats = model.statistics
            print(f"{i:2d}{'*' if model.is_loopless() else ' '} "
                  f"{model.get_name():20s} "
                  f"{model.level:2d} "
                  f"{stats['h']:8.4f} "
                  f"{stats['ddf']:4d} "
                  f"{stats['dlr']:8.4f} "
                  f"{stats['alpha']:8.4f} "
                  f"{stats['information']:8.4f} "
                  f"{stats['dh_dv']:8.4f} "
                  f"{stats['aic']:8.4f} "
                  f"{stats['bic']:8.4f} "
                  f"{model.incremental_alpha:8.4f} "
                  f"{model.progenitor_id if model.progenitor_id else 0:4d} "
                  f"{self._compute_pct_correct(model):8.4f} "
                  f"{100.0000:8.4f}")
                  
        # Print best models summary
        print("\nBest Model(s) by dBIC:")
        best_bic = max(all_models, key=lambda m: m.statistics['bic'])
        print(f"{best_bic.get_name()}: {best_bic.statistics['bic']:.4f}")
        
        print("\nBest Model(s) by dAIC:")
        best_aic = max(all_models, key=lambda m: m.statistics['aic'])
        print(f"{best_aic.get_name()}: {best_aic.statistics['aic']:.4f}")
        
        print("\nBest Model(s) by Information:")
        best_info = max(all_models, key=lambda m: m.statistics['information'])
        print(f"{best_info.get_name()}: {best_info.statistics['information']:.4f}")
        
        return all_models
    
    def _sort_models(self, models: List[Model], criterion: str) -> List[Model]:
        """Sort models exactly as C++ VBMManager does"""
        # Define key functions to match C++ ordering
        key_funcs = {
            'h': lambda m: (m.statistics['h'], m.get_name()),
            'information': lambda m: (-m.statistics['information'], m.statistics['h'], m.get_name()),
            'aic': lambda m: (-m.statistics['aic'], m.statistics['h'], m.get_name()),
            'bic': lambda m: (-m.statistics['bic'], m.statistics['h'], m.get_name())
        }
        
        # Get sort key function
        key_func = key_funcs.get(criterion, key_funcs['h'])
        
        # Sort using stable sort
        return sorted(models, key=key_func)

    def _make_independence_model(self) -> Model:
        """Create independence model (IV:Z)"""
        return self.make_model("IV:Z")
        
    def _generate_parents(self, model: Model) -> List[Model]:
        """Generate parent models one level up"""
        parents = []
        
        # Get reference model for statistics computation
        if not hasattr(self, '_ref_model') or self._ref_model is None:
            self._ref_model = self._make_independence_model()
        
        # For each component except IV
        for i, comp in enumerate(model.components):
            if set(comp) == set(self.ivs):
                continue
                
            # Try adding each unused variable
            for var in self.all_vars:
                if var not in comp:
                    new_comps = [c.copy() for c in model.components]
                    new_comps[i] = sorted(comp + [var])
                    
                    parent = Model(
                        new_comps,
                        self.all_vars,
                        self.dv,
                        self.state_space
                    )
                    
                    # Compute statistics before setting progenitor
                    self._compute_model_statistics(parent, self._ref_model)
                    parent.set_progenitor(model)
                    
                    parents.append(parent)
                    
        return parents

    def _compute_model_statistics(self, model: Model, ref_model: Model):
        """Compute all model statistics using Relations"""
        model.compute_model_statistics(ref_model)
        
    def fit(self, model_name: str) -> Dict:
        """
        Fit a specific model and return detailed statistics.
        Matches format from Occam manual v3.4.1.
        """
        # Create all models we'll need
        model = self.make_model(model_name)
        ref_bottom = self._make_independence_model()
        ref_top = Model(
            components=[sorted(self.ivs + [self.dv])],
            variables=self.all_vars,
            dv_col=self.dv,
            state_space=self.state_space
        )

        # Get all generated relations from model spec
        components = model.get_name().split(':')
        all_models = []
        
        # Always include IV:Z as first (independence) model
        indep_model = self._make_independence_model()
        indep_model.level = 0
        self._compute_model_statistics(indep_model, ref_bottom)
        all_models.append(indep_model)
        
        # Add single-predicting component models (IV:ApZ, IV:EdZ, IV:CZ)
        for var in self.ivs:
            single_comp = f"IV:{var}Z"
            if single_comp in components or any(var in comp for comp in components[1:]):
                m = self.make_model(single_comp)
                m.level = 1
                self._compute_model_statistics(m, ref_bottom)
                all_models.append(m)
                
        # Add double-predicting component models (IV:ApEdZ, IV:ApCZ, etc.)
        for i, var1 in enumerate(self.ivs):
            for var2 in self.ivs[i+1:]:
                double_comp = f"IV:{var1}{var2}Z"
                if double_comp in components or any(all(v in comp for v in [var1, var2]) 
                                                 for comp in components[1:]):
                    m = self.make_model(double_comp)
                    m.level = 2
                    self._compute_model_statistics(m, ref_bottom)
                    all_models.append(m)
                    
        # Add triple and higher component models if present
        if len(model.components) > 2:  # More than IV and one predicting component
            self._compute_model_statistics(model, ref_bottom)
            model.level = 3
            all_models.append(model)
            
        # Sort models by H (entropy)
        all_models.sort(key=lambda m: m.statistics['h'])
        
        # Assign IDs
        for i, m in enumerate(all_models, 1):
            m.id = i

        # Print model details table
        print("\nID MODEL Level H dDF dLR Alpha Inf %dH(DV) dAIC dBIC Inc.Alpha Prog. %C(Data) %cover")
        for m in all_models:
            stats = m.statistics
            print(f"{m.id:2d}{'*' if m.is_loopless() else ' '} "
                  f"{m.get_name():20s} "
                  f"{m.level:1d} "
                  f"{stats['h']:8.4f} "
                  f"{stats['ddf']:4d} "
                  f"{stats['dlr']:8.4f} "
                  f"{stats['alpha']:8.4f} "
                  f"{stats['information']:8.4f} "
                  f"{stats['dh_dv']:8.4f} "
                  f"{stats['aic']:8.4f} "
                  f"{stats['bic']:8.4f} "
                  f"{0.0000:8.4f} " # Inc.Alpha placeholder
                  f"{0:4d} "        # Prog. placeholder
                  f"{self._compute_pct_correct(m):8.4f} "
                  f"{100.0000:8.4f}")

        # Print model structure
        print(f"\nModel {model_name} (Directed System)")
        print(f"IV Component: {'; '.join(self.ivs)} ({'+'.join(self.ivs)})")
        for comp in model.components[1:]:  # Skip IV component
            print(f"Model Component: {'; '.join(comp)} ({'+'.join(comp)})")
                
        # Compute and print basic statistics
        self._compute_model_statistics(model, ref_bottom)
        df = model.compute_df()
        h = model.statistics['h']
        info = model.statistics['information']
        t = abs(model.statistics['dlr']/(2 * self.state_space.n * np.log(2)))
        
        print(f"\nDegrees of Freedom (DF): {df}")
        print(f"Loops: {'YES' if not model.is_loopless() else 'NO'}")
        print(f"Entropy(H): {h:.6f}")
        print(f"Information captured (%): {info*100:.4f}")
        print(f"Transmission (T): {t:.6f}")
        
        # Get fitted probabilities
        q = model.get_fitted_probs()
        
        # Print reference tables
        print("\nREFERENCE = TOP")
        self._print_ref_table(model, ref_top)
        
        print("\nREFERENCE = BOTTOM") 
        self._print_ref_table(model, ref_bottom)
        
        # Print conditional DV tables
        print("\nConditional DV (D) (%) for each IV composite state for the Model " + model_name + ".")
        print(f"IV order: {', '.join(model.components[0])} ({', '.join(self.ivs)})")
        self._print_conditional_table(model, q)
        
        # Print component tables if model has multiple components
        if len(model.components) > 2:  # More than IV and one predicting component
            for comp in model.components[1:]:
                print(f"\nConditional DV (D) (%) for each IV composite state for the Relation {''.join(comp)}.")
                print("(For component relations, the Data and Model parts of the table are equal, so only one is given)")
                comp_model = Model(
                    components=[self.ivs, comp],
                    variables=self.all_vars,
                    dv_col=self.dv,
                    state_space=self.state_space
                )
                comp_q = comp_model.get_fitted_probs()
                self._print_conditional_table(comp_model, comp_q, is_component=True)
                    
        return {
            'model': model,
            'statistics': model.statistics,
            'fitted_probs': q
        }
    def print_fit_report(self, model: Model):
        """Print detailed fit report"""
        print("\nModel Components:")
        for comp in model.components:
            print(f"  {' '.join(comp)} ({', '.join(self.varlist.get_full_name(v) for v in comp)})")

    def _compute_pct_correct(self, model: Model) -> float:
        """Compute percent correct predictions for a model"""
        q = model.get_fitted_probs()
        total_correct = 0
        total_cases = 0
        
        # Group states by IV values
        iv_groups = defaultdict(list)
        for state in q.keys():
            iv_state = state.project(self.ivs)
            iv_groups[iv_state].append(state)
            
        # For each IV state
        for iv_state in iv_groups:
            states = iv_groups[iv_state]
            freq = sum(self.state_space.state_frequencies[s] for s in states)
            
            if freq > 0:
                # Get predicted DV value (highest probability)
                dv_probs = defaultdict(float)
                for s in states:
                    dv_val = s.get_variable_values([self.dv])[0]
                    dv_probs[dv_val] += q[s]
                pred_dv = max(dv_probs.items(), key=lambda x: x[1])[0]
                
                # Count correct predictions
                correct = sum(self.state_space.state_frequencies[s] 
                            for s in states 
                            if s.get_variable_values([self.dv])[0] == pred_dv)
                
                total_correct += correct
                total_cases += freq
                
        return (total_correct / total_cases * 100) if total_cases > 0 else 0

    def _print_ref_table(self, model: Model, ref_model: Model):
        """Print reference model comparison table"""
        self._compute_model_statistics(model, ref_model)
        print("Value Prob. (Alpha)")
        print(f"Log-Likelihood (LR) {model.statistics['dlr']:8.4f} {model.statistics['alpha']:8.4f}")
        # Add Pearson chi-square when implemented
        print(f"Delta DF (dDF) {model.statistics['ddf']}")
        
    def _print_conditional_table(self, model: Model, q: Dict[State, float], is_component: bool = False):
        """Print conditional probability table"""
        print(f"IV order: {', '.join(model.components[0])}")  # Show IV order
        if is_component:
            print("(For component relations, the Data and Model parts of the table are equal, so only one is given)")
            print("IV| Data")
            print("| obs. p(DV|IV)")
        else:
            print("IV | Data | Model")
            print("| obs. p(DV|IV) | calc. q(DV|IV)")
            
        # Print variable abbreviations header
        iv_abbrev_line = ' '.join(model.components[0])
        print(f"{iv_abbrev_line:10s} | {'freq':6s} {'Z=0':6s} {'Z=1':6s}", end='')
        if not is_component:
            print(f" | {'Z=0':6s} {'Z=1':6s} {'rule':4s} {'#correct':8s} {'%correct':8s}")
        else:
            print()
            
        # Group states by IV values
        iv_states = defaultdict(list)
        for state in q.keys():
            iv_state = state.project(self.ivs)
            iv_states[iv_state].append(state)
            
        total_correct = 0
        total_cases = 0
        
        # For each IV state
        for iv_state in sorted(iv_states.keys()):
            # Get frequencies and probabilities
            states = iv_states[iv_state]
            freq = sum(self.state_space.state_frequencies[s] for s in states)
            
            # Calculate observed proportions and convert to percentages
            obs_probs = [0, 0]  # For DV=0 and DV=1
            for s in states:
                dv_val = s.get_variable_values([self.dv])[0]
                obs_probs[dv_val] += self.state_space.state_frequencies[s] / freq
            # Convert to percentages
            obs_probs = [p * 100 for p in obs_probs]
                
            # Print basic info
            iv_vals = [str(v) for v in iv_state.values]
            row = f"{' '.join(iv_vals):10s} | {freq:6.3f} {obs_probs[0]:6.3f} {obs_probs[1]:6.3f}"
            
            if not is_component:
                # Calculate model probabilities and convert to percentages
                model_probs = [0, 0]
                for s in states:
                    dv_val = s.get_variable_values([self.dv])[0]
                    model_probs[dv_val] += q[s] / sum(q[s2] for s2 in states)
                # Convert to percentages
                model_probs = [p * 100 for p in model_probs]
                    
                # Get prediction rule
                pred = 0 if model_probs[0] > model_probs[1] else 1
                
                # Calculate accuracy
                correct = sum(self.state_space.state_frequencies[s] 
                             for s in states 
                             if s.get_variable_values([self.dv])[0] == pred)
                pct_correct = (correct / freq) * 100 if freq > 0 else 0
                
                row += f" | {model_probs[0]:6.3f} {model_probs[1]:6.3f} {pred:4d} {correct:8.0f} {pct_correct:8.2f}"
                
                total_correct += correct
                total_cases += freq
                
            print(row)
            
        # Print summary row
        if not is_component:
            total_pct = (total_correct / total_cases) * 100 if total_cases > 0 else 0
            print(f"{'TOTAL':10s} | {total_cases:6.0f} | | | | {total_pct:8.2f}")
            print(f"| freq Z=0 Z=1| Z=0 Z=1rule#correct%correct")  # Column labels
        else:
            print(f"| freq Z=0 Z=1|")  # Column labels

    def _remove_duplicate_models(self, models: List[Model]) -> List[Model]:
        """
        Remove duplicate models from list.
        A model is considered duplicate if it is equivalent to an existing model.
        
        Args:
            models: List of candidate models
            
        Returns:
            List with duplicates removed
        """
        unique_models = []
        seen = set()  # Track model signatures 
        
        for model in models:
            # Create signature from sorted components
            signature = tuple(sorted(
                tuple(sorted(comp)) for comp in model.components
            ))
            
            # Only keep if we haven't seen this signature
            if signature not in seen:
                seen.add(signature)
                unique_models.append(model)
                
        return unique_models        
        
    def _compute_predictions(self, model: Model, q: Dict[State, float]) -> Dict[str, Dict]:
        """
        Compute predictions for each IV state based on fitted probabilities.
        
        Args:
            model: Model to compute predictions for
            q: Fitted probabilities from IPF
            
        Returns:
            Dictionary with:
                - iv_states: Dict mapping IV state to predicted DV state
                - accuracies: Dict mapping IV state to prediction accuracy
                - rules: Dict mapping IV state to rule source
        """
        predictions = {
            'iv_states': {},    # Predicted DV state for each IV state
            'accuracies': {},   # Accuracy of prediction
            'rules': {}         # Source of prediction rule 
        }
        
        # Get independence model probabilities for backup predictions
        indep_model = self._make_independence_model()
        indep_q = indep_model.get_fitted_probs()
        
        # Group states by IV values
        iv_groups = defaultdict(list)
        for state in q.keys():
            # Project state to just IVs
            iv_state = state.project(self.ivs)
            iv_groups[iv_state].append(state)
            
        # For each IV state
        for iv_state in iv_groups:
            # Get conditional DV probabilities
            dv_probs = defaultdict(float)
            total = 0.0
            
            # Sum probabilities for each DV value
            for full_state in iv_groups[iv_state]:
                dv_value = full_state.get_variable_values([self.dv])[0]
                dv_probs[dv_value] += q[full_state]
                total += q[full_state]
                
            # Normalize probabilities
            if total > 0:
                dv_probs = {k: v/total for k, v in dv_probs.items()}
                
                # Find DV value with highest probability
                pred_dv = max(dv_probs.items(), key=lambda x: x[1])[0]
                predictions['iv_states'][iv_state] = pred_dv
                predictions['accuracies'][iv_state] = dv_probs[pred_dv]
                predictions['rules'][iv_state] = 'model'
                
            else:
                # If no probability mass, use independence model
                dv_probs = defaultdict(float)
                for state, prob in indep_q.items():
                    if state.project(self.ivs) == iv_state:
                        dv_value = state.get_variable_values([self.dv])[0]
                        dv_probs[dv_value] += prob
                        
                if dv_probs:
                    pred_dv = max(dv_probs.items(), key=lambda x: x[1])[0]
                    predictions['iv_states'][iv_state] = pred_dv
                    predictions['accuracies'][iv_state] = dv_probs[pred_dv]
                    predictions['rules'][iv_state] = 'independence'
                    
        return predictions        