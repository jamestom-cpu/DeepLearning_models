from typing import Callable, Tuple, List, Dict



def check_filter_format(filters, verbose=False):

        def mprint(*args):
            if verbose:
                print(*args)
        # Check that the filters are lists of tuples of functions and tuples of arguments
        # convert other input types to this format

        # check for None
        if filters is None or filters==() or filters==(None, ):
            mprint("None")
            return []
        # check if already in correct format
        if isinstance(filters, List):
            if all([
                isinstance(filter, Tuple) and
                len(filter)==3 and 
                isinstance(filter[0], Callable) and 
                isinstance(filter[1], tuple) and 
                isinstance(filter[2], dict) 
                for filter in filters]):
                mprint("already in correct form")
                return filters




        if isinstance(filters, Callable):
            mprint("A0")
            filters = [(filters, ())]
        if isinstance(filters, Tuple):
            mprint("B0")
            filters = [filters]
        if isinstance(filters, List):
            for i, filter in enumerate(filters):
                mprint("A1")
                if isinstance(filter, Callable):
                    filters[i] = (filter, ())
                    mprint("A")
                elif isinstance(filter, Tuple) and isinstance(filter[0], Callable):
                    mprint("C")
                    if  len(filter)==2 and not isinstance(filter[1], Tuple):
                        mprint("D")
                        filters[i] = (filter[0], (filter[1],))
                    elif len(filter)==2 and isinstance(filter[1], Tuple):
                        mprint("E")
                        pass
                    elif len(filter)>2:
                        mprint("F")
                        filters[i] = (filter[0], tuple(filter[1:]))
                else:
                    raise ValueError("filters must be lists of tuples of functions and tuples of arguments")
        else:
            raise ValueError("filters must be lists of tuples of functions and tuples of arguments")
        
        # split in args and kwargs
        for i, filter in enumerate(filters):
            mprint("B1")
            arguments = filter[1]
            kwargs = {}
            args = ()
            for a in arguments:
                if isinstance(a, dict):
                    kwargs.update(a)
                else:
                    args += (a,)
            filters[i] = (filter[0], args, kwargs)
        return filters